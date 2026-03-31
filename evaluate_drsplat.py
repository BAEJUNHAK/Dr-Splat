"""
Dr. Splat 정량 평가 스크립트 (Ref-lerf 포맷 지원)
GT 마스크(PNG)와 Dr. Splat activation 결과를 비교하여 mIoU 등을 계산

Ref-lerf 데이터 구조:
    <ref_lerf_dir>/<scene>/json/test_json/frame_XXXXX.json
    <ref_lerf_dir>/<scene>/gt_mask/*.png

Usage:
    python evaluate_drsplat.py \
        -s <scene_path> \
        -m <model_path> \
        --pq_index <pq_index_path> \
        --ref_lerf_dir <ref_lerf_dir> \
        --output_dir <output_dir> \
        --threshold 0.5
"""

import os
import re
import json
import numpy as np
import torch
import cv2
import faiss
from argparse import ArgumentParser
from tqdm import tqdm

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from evaluation.openclip_encoder import OpenCLIPNetwork


def compute_metrics(pred_mask, gt_mask):
    """IoU, Precision, Recall, F1 계산"""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_area = pred.sum()
    gt_area = gt.sum()
    total_pixels = gt_mask.shape[0] * gt_mask.shape[1]

    iou = (intersection / union * 100) if union > 0 else 0.0
    precision = (intersection / pred_area * 100) if pred_area > 0 else 0.0
    recall = (intersection / gt_area * 100) if gt_area > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    detected = 1 if intersection > 0 else 0
    acc_25 = 1 if iou > 25.0 else 0
    area_ratio = (gt_area / total_pixels * 100)

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detected": detected,
        "acc_25": acc_25,
        "gt_area": int(gt_area),
        "pred_area": int(pred_area),
        "area_ratio": area_ratio,
    }


def render_activation_mask(gaussians, view, activation_features, threshold, pipeline, background, args):
    """activation을 기반으로 바이너리 마스크를 2D 렌더링"""
    activation_mask = activation_features.squeeze() > threshold

    saved_dc = gaussians._features_dc.data.clone()
    saved_rest = gaussians._features_rest.data.clone()

    white_sh = (torch.tensor([[[1.0, 1.0, 1.0]]], device="cuda") - 0.5) / 0.28209479177387814
    black_sh = (torch.tensor([[[0.0, 0.0, 0.0]]], device="cuda") - 0.5) / 0.28209479177387814

    gaussians._features_dc.data[activation_mask] = white_sh.expand(activation_mask.sum(), 1, 3)
    gaussians._features_rest.data[activation_mask] = 0
    gaussians._features_dc.data[~activation_mask] = black_sh.expand((~activation_mask).sum(), 1, 3)
    gaussians._features_rest.data[~activation_mask] = 0

    output = render(view, gaussians, pipeline, background, args)
    rendered = output["render"]

    gaussians._features_dc.data.copy_(saved_dc)
    gaussians._features_rest.data.copy_(saved_rest)

    gray = rendered.mean(dim=0).cpu().numpy()
    binary_mask = (gray > 0.5).astype(np.uint8)

    return binary_mask, rendered


def main():
    parser = ArgumentParser(description="Dr. Splat Evaluation (Ref-lerf format)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--pq_index", type=str, required=True)
    parser.add_argument("--ref_lerf_dir", type=str, required=True,
                        help="Path to Ref-lerf root (contains <scene>/json/test_json/ and <scene>/gt_mask/)")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--custom_queries", type=str, default=None,
                        help="Path to custom queries JSON. Format: [{\"query\": \"yellow bowl\", \"gt_category\": \"bowl\", \"level\": \"L1\"}, ...]")

    args = get_combined_args(parser)

    device = "cuda"
    clip_model = OpenCLIPNetwork(device)
    index = faiss.read_index(args.pq_index)

    # 모델 로드
    gaussians = GaussianModel(3)
    dataset = model.extract(args)
    scene = Scene(dataset, gaussians, shuffle=False)

    checkpoint = os.path.join(args.model_path, 'chkpnt0.pth')
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
    gaussians.restore(model_params, args, mode='test')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    pipe = pipeline.extract(args)

    # PQ 디코딩
    features = gaussians._language_feature.clone()
    zero_mask = torch.all(features == -1, dim=-1)
    leaf_lang_feat = torch.from_numpy(index.sa_decode(features[~zero_mask].cpu().numpy())).to(device)

    # Ref-lerf 경로 구성
    scene_name = os.path.basename(dataset.source_path)
    ref_lerf_scene = os.path.join(args.ref_lerf_dir, scene_name)
    test_json_dir = os.path.join(ref_lerf_scene, "json", "test_json")
    # Ref-lerf 실제 폴더명은 "mask" (gt_mask 아님)
    gt_mask_dir = os.path.join(ref_lerf_scene, "mask")
    if not os.path.isdir(gt_mask_dir):
        gt_mask_dir = os.path.join(ref_lerf_scene, "gt_mask")  # fallback

    if not os.path.isdir(test_json_dir):
        print(f"[ERROR] test_json dir not found: {test_json_dir}")
        return
    if not os.path.isdir(gt_mask_dir):
        print(f"[ERROR] gt_mask dir not found: {gt_mask_dir}")
        return

    json_files = sorted([f for f in os.listdir(test_json_dir) if f.endswith('.json')])
    print(f"  Scene: {scene_name}")
    print(f"  Test JSON files: {len(json_files)}")
    print(f"  GT mask dir: {gt_mask_dir}")

    # 카메라 뷰 (train + test 모두에서 매칭)
    all_cameras = scene.getTrainCameras() + scene.getTestCameras()
    cam_dict = {}
    for cam in all_cameras:
        cam_dict[cam.image_name] = cam
    print(f"  Total cameras: {len(cam_dict)} (train: {len(scene.getTrainCameras())}, test: {len(scene.getTestCameras())})")

    output_dir = os.path.join(args.output_dir, scene_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pred_masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "gt_masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlay"), exist_ok=True)

    all_results = []

    for json_file in tqdm(json_files, desc="Evaluating"):
        json_path = os.path.join(test_json_dir, json_file)
        with open(json_path) as f:
            data = json.load(f)

        frame_name = json_file.replace('.json', '')

        # 카메라 뷰 찾기
        cam = cam_dict.get(frame_name)
        if cam is None:
            frame_nums = re.findall(r'\d+', frame_name)
            frame_num = frame_nums[-1] if frame_nums else None
            if frame_num:
                for cam_name, cam_obj in cam_dict.items():
                    cam_nums = re.findall(r'\d+', cam_name)
                    if cam_nums and cam_nums[-1] == frame_num:
                        cam = cam_obj
                        break
        if cam is None:
            print(f"[WARN] Camera not found for {frame_name}, skipping")
            continue
        print(f"  {frame_name} -> {cam.image_name}")

        # 원본 이미지 렌더링 (오버레이용)
        with torch.no_grad():
            orig_output = render(cam, gaussians, pipe, background, args)
            orig_img = orig_output["render"].cpu().permute(1, 2, 0).numpy()
            orig_img = np.clip(orig_img * 255, 0, 255).astype(np.uint8)
            render_h, render_w = orig_img.shape[:2]

        # Ref-lerf JSON: "object" (not "objects")
        objects = data.get("object", data.get("objects", []))

        for obj in objects:
            category = obj.get("category", "unknown")
            segmentation = obj.get("segmentation", "")
            sentences = obj.get("sentence", [])
            # 이전 포맷 호환: "note" 필드
            if not sentences and "note" in obj:
                sentences = [obj["note"]]

            # GT 마스크 로드 (PNG 파일 경로 or 폴리곤 좌표)
            if isinstance(segmentation, str) and segmentation:
                # Ref-lerf 포맷: segmentation = "frame_00006/bowl.png" 등
                # 여러 경로 후보 시도
                candidates = [
                    os.path.join(gt_mask_dir, segmentation),                    # gt_mask/frame_00006/bowl.png
                    os.path.join(ref_lerf_scene, segmentation),                 # ramen/frame_00006/bowl.png
                    os.path.join(ref_lerf_scene, "gt_mask", segmentation),      # ramen/gt_mask/frame_00006/bowl.png
                    os.path.join(gt_mask_dir, os.path.basename(segmentation)),   # gt_mask/bowl.png
                ]
                mask_path = None
                for c in candidates:
                    if os.path.exists(c):
                        mask_path = c
                        break
                if mask_path is None:
                    print(f"    [WARN] GT mask not found: {segmentation}, tried: {candidates[0]}, skipping")
                    continue
                gt_mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask_full is None:
                    print(f"    [WARN] Failed to read mask: {mask_path}, skipping")
                    continue
                gt_mask_full = (gt_mask_full > 0).astype(np.uint8)
            elif isinstance(segmentation, list) and len(segmentation) > 0:
                # 이전 포맷: 폴리곤 좌표
                img_w = data.get("info", {}).get("width", render_w)
                img_h = data.get("info", {}).get("height", render_h)
                pts = np.array(segmentation, dtype=np.int32).reshape(-1, 1, 2)
                gt_mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.fillPoly(gt_mask_full, [pts], 1)
            else:
                print(f"    [WARN] Invalid segmentation for {category}, skipping")
                continue

            # 렌더링 해상도에 맞게 리사이즈
            gt_mask = cv2.resize(gt_mask_full, (render_w, render_h), interpolation=cv2.INTER_NEAREST)

            # 쿼리: (1) 카테고리 단어, (2) 설명 문장 1개 (첫 번째)
            queries = [("category", category)]
            if sentences:
                sent = sentences[0]
                if sent and sent != category:
                    queries.append(("sentence", sent))

            for query_type, query_text in queries:
                # CLIP 텍스트 인코딩 + activation 계산
                clip_model.set_positives([query_text])
                activation_features = torch.zeros((features.shape[0], 1), dtype=torch.float32).cuda()
                with torch.no_grad():
                    _activation = clip_model.get_activation(leaf_lang_feat, 0)
                activation_features[~zero_mask] = _activation

                # 2D 마스크 렌더링
                with torch.no_grad():
                    pred_mask, rendered = render_activation_mask(
                        gaussians, cam, activation_features, args.threshold,
                        pipe, background, args
                    )

                # 메트릭 계산
                metrics = compute_metrics(pred_mask, gt_mask)
                metrics["frame"] = frame_name
                metrics["category"] = category
                metrics["query_type"] = query_type
                metrics["query_text"] = query_text
                metrics["sentences"] = sentences
                metrics["scene"] = scene_name

                all_results.append(metrics)

                # 마스크 저장
                safe_cat = category.replace(" ", "_").replace("/", "_")
                safe_qt = query_type[0]  # 'c' or 's'
                pred_path = os.path.join(output_dir, "pred_masks", f"{frame_name}_{safe_cat}_{safe_qt}.png")
                gt_path = os.path.join(output_dir, "gt_masks", f"{frame_name}_{safe_cat}.png")
                cv2.imwrite(pred_path, pred_mask * 255)
                cv2.imwrite(gt_path, gt_mask * 255)

                # 오버레이 저장
                overlay = orig_img.copy().astype(np.float32)
                overlay[gt_mask == 1] = overlay[gt_mask == 1] * 0.6 + np.array([0, 255, 0], dtype=np.float32) * 0.4
                overlay[pred_mask == 1] = overlay[pred_mask == 1] * 0.7 + np.array([255, 0, 0], dtype=np.float32) * 0.3
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                overlay_path = os.path.join(output_dir, "overlay", f"{frame_name}_{safe_cat}_{safe_qt}_iou{metrics['iou']:.1f}.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                metrics["pred_path"] = pred_path
                metrics["gt_path"] = gt_path
                metrics["overlay_path"] = overlay_path

    # 커스텀 쿼리 평가 (--custom_queries)
    if args.custom_queries and os.path.exists(args.custom_queries):
        with open(args.custom_queries) as f:
            custom_queries = json.load(f)
        print(f"\n  Custom queries: {len(custom_queries)} from {args.custom_queries}")

        # 프레임별 GT 마스크 캐시: {frame_name: {category: gt_mask}}
        gt_cache = {}
        for json_file in sorted(os.listdir(test_json_dir)):
            if not json_file.endswith('.json'):
                continue
            frame_name = json_file.replace('.json', '')
            with open(os.path.join(test_json_dir, json_file)) as f:
                data = json.load(f)

            cam = cam_dict.get(frame_name)
            if cam is None:
                frame_nums = re.findall(r'\d+', frame_name)
                frame_num = frame_nums[-1] if frame_nums else None
                if frame_num:
                    for cam_name, cam_obj in cam_dict.items():
                        cam_nums = re.findall(r'\d+', cam_name)
                        if cam_nums and cam_nums[-1] == frame_num:
                            cam = cam_obj
                            break
            if cam is None:
                continue

            with torch.no_grad():
                orig_output = render(cam, gaussians, pipe, background, args)
                render_h, render_w = orig_output["render"].shape[1], orig_output["render"].shape[2]

            gt_cache[frame_name] = {"cam": cam, "masks": {}, "render_hw": (render_h, render_w)}
            objects = data.get("object", data.get("objects", []))
            for obj in objects:
                category = obj.get("category", "unknown")
                segmentation = obj.get("segmentation", "")
                if isinstance(segmentation, str) and segmentation:
                    candidates = [
                        os.path.join(gt_mask_dir, segmentation),
                        os.path.join(ref_lerf_scene, segmentation),
                        os.path.join(ref_lerf_scene, "gt_mask", segmentation),
                        os.path.join(gt_mask_dir, os.path.basename(segmentation)),
                    ]
                    mask_path = None
                    for c in candidates:
                        if os.path.exists(c):
                            mask_path = c
                            break
                    if mask_path is None:
                        continue
                    gt_mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if gt_mask_full is None:
                        continue
                    gt_mask_full = (gt_mask_full > 0).astype(np.uint8)
                    gt_mask_resized = cv2.resize(gt_mask_full, (render_w, render_h), interpolation=cv2.INTER_NEAREST)
                    gt_cache[frame_name]["masks"][category] = gt_mask_resized

        for cq in tqdm(custom_queries, desc="Custom queries"):
            query_text = cq["query"]
            gt_category = cq["gt_category"]
            level = cq.get("level", "custom")

            clip_model.set_positives([query_text])
            activation_features = torch.zeros((features.shape[0], 1), dtype=torch.float32).cuda()
            with torch.no_grad():
                _activation = clip_model.get_activation(leaf_lang_feat, 0)
            activation_features[~zero_mask] = _activation

            for frame_name, cache in gt_cache.items():
                if gt_category not in cache["masks"]:
                    continue
                gt_mask = cache["masks"][gt_category]
                cam = cache["cam"]

                with torch.no_grad():
                    pred_mask, rendered = render_activation_mask(
                        gaussians, cam, activation_features, args.threshold,
                        pipe, background, args
                    )

                metrics = compute_metrics(pred_mask, gt_mask)
                metrics["frame"] = frame_name
                metrics["category"] = gt_category
                metrics["query_type"] = level
                metrics["query_text"] = query_text
                metrics["sentences"] = []
                metrics["scene"] = scene_name

                safe_label = query_text.replace(" ", "_")[:40]
                safe_cat = gt_category.replace(" ", "_").replace("/", "_")
                pred_path = os.path.join(output_dir, "pred_masks", f"{frame_name}_{safe_cat}_{safe_label}.png")
                cv2.imwrite(pred_path, pred_mask * 255)
                metrics["pred_path"] = pred_path
                metrics["gt_path"] = os.path.join(output_dir, "gt_masks", f"{frame_name}_{safe_cat}.png")

                all_results.append(metrics)

    # 결과 저장
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 요약 출력
    if all_results:
        from collections import defaultdict

        ious = [r["iou"] for r in all_results]
        precisions = [r["precision"] for r in all_results]
        recalls = [r["recall"] for r in all_results]
        f1s = [r["f1"] for r in all_results]
        detected = [r["detected"] for r in all_results]
        acc_25s = [r["acc_25"] for r in all_results]

        print(f"\n{'='*60}")
        print(f"Dr. Splat Evaluation Results - {scene_name}")
        print(f"{'='*60}")
        print(f"  Total queries:    {len(all_results)}")
        print(f"  mIoU:             {np.mean(ious):.2f}%")
        print(f"  mAcc@0.25:        {np.mean(acc_25s)*100:.2f}%")
        print(f"  Precision:        {np.mean(precisions):.2f}%")
        print(f"  Recall:           {np.mean(recalls):.2f}%")
        print(f"  F1:               {np.mean(f1s):.2f}%")
        print(f"  Detection Rate:   {np.mean(detected)*100:.1f}%")
        print(f"{'='*60}")

        # 쿼리 타입별 출력 (category, sentence, L0, L1, L5, ...)
        query_types = sorted(set(r.get("query_type", "") for r in all_results))
        for qt in query_types:
            qt_results = [r for r in all_results if r.get("query_type") == qt]
            if not qt_results:
                continue
            qt_ious = [r["iou"] for r in qt_results]
            qt_acc25 = [r["acc_25"] for r in qt_results]
            print(f"\n--- Query Type: {qt} ---")
            print(f"  mIoU: {np.mean(qt_ious):.2f}%  mAcc@0.25: {np.mean(qt_acc25)*100:.2f}%  (n={len(qt_results)})")

            cat_results = defaultdict(list)
            cat_acc25 = defaultdict(list)
            for r in qt_results:
                cat_results[r["category"]].append(r["iou"])
                cat_acc25[r["category"]].append(r["acc_25"])

            print(f"\n  {'Category':<25s} {'mIoU':>8s} {'Acc@.25':>8s} {'Count':>6s}")
            print("  " + "-" * 51)
            for cat in sorted(cat_results.keys()):
                ious_cat = cat_results[cat]
                acc_cat = cat_acc25[cat]
                print(f"    {cat:<23s} {np.mean(ious_cat):>7.2f}% {np.mean(acc_cat)*100:>7.2f}% {len(ious_cat):>5d}")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
