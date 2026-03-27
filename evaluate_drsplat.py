"""
Dr. Splat 정량 평가 스크립트
GT 마스크(JSON 폴리곤)와 Dr. Splat activation 결과를 비교하여 mIoU 등을 계산

Usage:
    python evaluate_drsplat.py \
        -s <scene_path> \
        -m <model_path> \
        --pq_index <pq_index_path> \
        --label_dir <label_dir> \
        --output_dir <output_dir> \
        --threshold 0.5
"""

import os
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


def polygon_to_mask(polygon, height, width):
    """JSON 폴리곤 좌표를 바이너리 마스크로 변환"""
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


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
    area_ratio = (gt_area / total_pixels * 100)

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detected": detected,
        "gt_area": int(gt_area),
        "pred_area": int(pred_area),
        "area_ratio": area_ratio,
    }


def render_activation_mask(gaussians, view, activation_features, threshold, pipeline, background, args,
                           orig_dc=None, orig_rest=None):
    """activation을 기반으로 바이너리 마스크를 2D 렌더링 (deepcopy 대신 색상만 교체)"""
    activation_mask = activation_features.squeeze() > threshold

    # 기존 색상 백업 (최초 1회만)
    saved_dc = gaussians._features_dc.data.clone()
    saved_rest = gaussians._features_rest.data.clone()

    # 활성화된 Gaussian = 흰색, 나머지 = 검정
    white_sh = (torch.tensor([[[1.0, 1.0, 1.0]]], device="cuda") - 0.5) / 0.28209479177387814
    black_sh = (torch.tensor([[[0.0, 0.0, 0.0]]], device="cuda") - 0.5) / 0.28209479177387814

    gaussians._features_dc.data[activation_mask] = white_sh.expand(activation_mask.sum(), 1, 3)
    gaussians._features_rest.data[activation_mask] = 0
    gaussians._features_dc.data[~activation_mask] = black_sh.expand((~activation_mask).sum(), 1, 3)
    gaussians._features_rest.data[~activation_mask] = 0

    output = render(view, gaussians, pipeline, background, args)
    rendered = output["render"]  # [3, H, W]

    # 색상 복원
    gaussians._features_dc.data.copy_(saved_dc)
    gaussians._features_rest.data.copy_(saved_rest)

    # grayscale로 변환 후 threshold
    gray = rendered.mean(dim=0).cpu().numpy()  # [H, W]
    binary_mask = (gray > 0.5).astype(np.uint8)

    return binary_mask, rendered


def main():
    parser = ArgumentParser(description="Dr. Splat Evaluation")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--pq_index", type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")

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

    # PQ 디코딩: 3D Gaussian의 language feature
    features = gaussians._language_feature.clone()
    zero_mask = torch.all(features == -1, dim=-1)
    leaf_lang_feat = torch.from_numpy(index.sa_decode(features[~zero_mask].cpu().numpy())).to(device)

    # GT label 로드
    label_dir = args.label_dir
    scene_name = os.path.basename(dataset.source_path)
    scene_label_dir = os.path.join(label_dir, scene_name)

    if not os.path.exists(scene_label_dir):
        print(f"[ERROR] Label dir not found: {scene_label_dir}")
        return

    # 테스트 프레임 매칭
    json_files = sorted([f for f in os.listdir(scene_label_dir) if f.endswith('.json')])

    # 카메라 뷰 가져오기 (train + test)
    all_cameras = scene.getTrainCameras() + scene.getTestCameras()
    cam_dict = {}
    for cam in all_cameras:
        cam_dict[cam.image_name] = cam

    output_dir = os.path.join(args.output_dir, scene_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pred_masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "gt_masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlay"), exist_ok=True)

    all_results = []

    for json_file in tqdm(json_files, desc="Evaluating"):
        json_path = os.path.join(scene_label_dir, json_file)
        with open(json_path) as f:
            data = json.load(f)

        frame_name = json_file.replace('.json', '')
        img_w = data["info"]["width"]
        img_h = data["info"]["height"]

        # 카메라 뷰 찾기 (정확 매칭 → 숫자 매칭 → 서브스트링)
        cam = cam_dict.get(frame_name)
        if cam is None:
            # frame_00060에서 숫자 추출하여 매칭
            import re
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

        # 원본 이미지 렌더링 (오버레이용)
        with torch.no_grad():
            orig_output = render(cam, gaussians, pipe, background, args)
            orig_img = orig_output["render"].cpu().permute(1, 2, 0).numpy()
            orig_img = np.clip(orig_img * 255, 0, 255).astype(np.uint8)
            render_h, render_w = orig_img.shape[:2]

        for obj in data["objects"]:
            category = obj["category"]
            segmentation = obj["segmentation"]
            sentence = obj.get("note", "")

            # GT 마스크 생성
            gt_mask_full = polygon_to_mask(segmentation, img_h, img_w)
            gt_mask = cv2.resize(gt_mask_full, (render_w, render_h), interpolation=cv2.INTER_NEAREST)

            # 두 가지 쿼리로 평가: (1) 카테고리 단어, (2) 설명 문장
            queries = [("category", category)]
            if sentence and sentence != category:
                queries.append(("sentence", sentence))

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
                metrics["sentence"] = sentence
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

    # 결과 저장
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # 요약 출력
    if all_results:
        ious = [r["iou"] for r in all_results]
        precisions = [r["precision"] for r in all_results]
        recalls = [r["recall"] for r in all_results]
        f1s = [r["f1"] for r in all_results]
        detected = [r["detected"] for r in all_results]

        print(f"\n{'='*60}")
        print(f"Dr. Splat Evaluation Results - {scene_name}")
        print(f"{'='*60}")
        print(f"  Total queries:    {len(all_results)}")
        print(f"  mIoU:             {np.mean(ious):.2f}%")
        print(f"  Precision:        {np.mean(precisions):.2f}%")
        print(f"  Recall:           {np.mean(recalls):.2f}%")
        print(f"  F1:               {np.mean(f1s):.2f}%")
        print(f"  Detection Rate:   {np.mean(detected)*100:.1f}%")
        print(f"{'='*60}")

        # query_type별 결과
        from collections import defaultdict
        for qt in ["category", "sentence"]:
            qt_results = [r for r in all_results if r.get("query_type") == qt]
            if not qt_results:
                continue
            qt_ious = [r["iou"] for r in qt_results]
            print(f"\n--- Query Type: {qt} ---")
            print(f"  mIoU: {np.mean(qt_ious):.2f}%  (n={len(qt_results)})")

            cat_results = defaultdict(list)
            for r in qt_results:
                cat_results[r["category"]].append(r["iou"])

            print(f"\n  {'Category':<25s} {'mIoU':>8s} {'Count':>6s}")
            print("  " + "-" * 43)
            for cat in sorted(cat_results.keys()):
                ious_cat = cat_results[cat]
                print(f"    {cat:<23s} {np.mean(ious_cat):>7.2f}% {len(ious_cat):>5d}")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
