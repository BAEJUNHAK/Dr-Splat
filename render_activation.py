#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
from scene import Scene
import os

from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from pathlib import Path

from evaluation.openclip_encoder import OpenCLIPNetwork

import time

import faiss

from evaluation import colormaps

import cv2

COLORMAP_OPTIONS = colormaps.ColormapOptions(
    colormap="turbo",
    normalize=True,
    colormap_min=-1.0,
    colormap_max=1.0,
)

def render_set(model_path, source_path, name, iteration, views, gaussians, gaussians_orig, pipeline, background, args, label, clip_model, img_label, activation_features, thr):
    render_path = os.path.join(model_path, name, f"renders_colormap_{img_label}")
    binary_path = os.path.join(model_path, name, f"renders_binary_{img_label}")
    sidebyside_path = os.path.join(model_path, name, f"renders_sidebyside_{img_label}")
    makedirs(render_path, exist_ok=True)
    makedirs(binary_path, exist_ok=True)
    makedirs(sidebyside_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # activation colormap 렌더링
        output = render(view, gaussians, pipeline, background, args)
        rendering = output["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))

        # 원본 이미지 렌더링
        output_orig = render(view, gaussians_orig, pipeline, background, args)
        rendering_orig = output_orig["render"]

        # 바이너리 마스크: activation을 렌더링해서 2D 마스크 생성
        # activation_features를 language_feature 자리에 넣어서 렌더링
        binary_gaussians = GaussianModel(gaussians_orig.active_sh_degree)
        binary_gaussians._xyz = gaussians_orig._xyz
        binary_gaussians._features_dc = gaussians_orig._features_dc.clone()
        binary_gaussians._features_rest = gaussians_orig._features_rest.clone()
        binary_gaussians._scaling = gaussians_orig._scaling
        binary_gaussians._rotation = gaussians_orig._rotation
        binary_gaussians._opacity = gaussians_orig._opacity
        binary_gaussians.active_sh_degree = gaussians_orig.active_sh_degree

        # threshold 이상인 Gaussian만 빨간색으로
        activation_mask = activation_features.squeeze() > thr
        red_color = torch.tensor([[[1.0, 0.0, 0.0]]], device="cuda")
        red_sh = (red_color - 0.5) / 0.28209479177387814
        binary_gaussians._features_dc[activation_mask] = red_sh.expand(activation_mask.sum(), 1, 3)
        binary_gaussians._features_rest[activation_mask] = 0

        # 비활성 Gaussian은 어둡게
        gray_color = torch.tensor([[[0.3, 0.3, 0.3]]], device="cuda")
        gray_sh = (gray_color - 0.5) / 0.28209479177387814
        binary_gaussians._features_dc[~activation_mask] = gray_sh.expand((~activation_mask).sum(), 1, 3)
        binary_gaussians._features_rest[~activation_mask] = 0

        output_binary = render(view, binary_gaussians, pipeline, background, args)
        rendering_binary = output_binary["render"]
        torchvision.utils.save_image(rendering_binary, os.path.join(binary_path, f"{idx:05d}.png"))

        # side-by-side: 원본 | colormap | 바이너리
        combined = torch.cat([rendering_orig, rendering, rendering_binary], dim=2)  # 가로로 이어붙임
        torchvision.utils.save_image(combined, os.path.join(sidebyside_path, f"{idx:05d}.png"))


def render_sets(dataset : ModelParams, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args, label, clip_model, index, img_save_label):
    with torch.no_grad():
        start_time = time.time()

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)

        checkpoint = os.path.join(args.model_path, 'chkpnt0.pth')

        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, args, mode='test')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 원본 Gaussian 보존 (side-by-side용)
        import copy
        gaussians_orig = copy.deepcopy(gaussians)

        features = gaussians._language_feature.clone()
        zero_mask = torch.all(features == -1, dim=-1)

        leaf_lang_feat = torch.from_numpy(index.sa_decode(features[~zero_mask].cpu().numpy())).to("cuda")
        activation_features = torch.zeros((features.shape[0], 1), dtype=torch.float32).cuda()
        _activation_features = clip_model.get_activation(leaf_lang_feat, label)
        activation_features[~zero_mask] = _activation_features

        thr = args.threshold

        activation_threshold = torch.where(activation_features.squeeze() > thr)[0]

        features_colormap = colormaps.apply_colormap(activation_features, colormap_options=COLORMAP_OPTIONS)
        features_colormap = (features_colormap.unsqueeze(1) - 0.5) / 0.28209479177387814
        gaussians._features_dc[activation_threshold] = features_colormap[activation_threshold]
        gaussians._features_rest[activation_threshold] = torch.zeros_like(gaussians._features_rest)[activation_threshold].cuda()


        end_time = time.time()
        print(f'Running time : {end_time - start_time}')

        if not skip_train:
             render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, gaussians_orig, pipeline, background, args, label, clip_model, img_save_label, activation_features, thr)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, gaussians_orig, pipeline, background, args, label, clip_model, img_save_label, activation_features, thr)

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--semantic_model", default='dino', type=str)
    parser.add_argument("--pq_index", type=str, default=None)
    parser.add_argument("--img_save_label", type=str, default=None)
    parser.add_argument("--img_label", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.0)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    
    img_labels = [args.img_label]

    device = "cuda"
    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives(img_labels)


    index = faiss.read_index(args.pq_index)

    negative_text_features = torch.from_numpy(np.load('assets/text_negative.npy')).to(torch.float32)  # [num_text, 512]

    for label in range(len(img_labels)):
        text_feat = clip_model.encode_text(img_labels[label], device=device).float()
        render_sets(model.extract(args), pipeline.extract(args), args.skip_train, args.skip_test, args, label, clip_model, index, args.img_save_label)