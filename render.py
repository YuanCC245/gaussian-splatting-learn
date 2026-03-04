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
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    """
    渲染指定视角集合并保存结果

    参数:
        model_path: 模型路径
        name: 视角集合名称("train" 或 "test")
        iteration: 要渲染的模型迭代次数
        views: 要渲染的相机视角列表
        gaussians: 高斯模型对象
        pipeline: 渲染管线参数
        background: 背景颜色
        train_test_exp: 是否使用训练/测试曝光
        separate_sh: 是否分离球谐系数(用于 SparseAdam)
    """
    # 创建渲染输出目录
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # 遍历每个视角进行渲染
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 渲染当前视角
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        # 获取对应的真实图像(ground truth)
        gt = view.original_image[0:3, :, :]

        # 如果使用了曝光补偿,只使用右半部分图像
        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        # 保存渲染图像和真实图像
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    """
    渲染训练集和测试集

    参数:
        dataset: 模型参数(数据集路径、分辨率等)
        iteration: 要加载的模型迭代次数(-1 表示最新)
        pipeline: 渲染管线参数
        skip_train: 是否跳过训练集渲染
        skip_test: 是否跳过测试集渲染
        separate_sh: 是否分离球谐系数
    """
    with torch.no_grad():
        # 创建高斯模型
        gaussians = GaussianModel(dataset.sh_degree)
        # 加载场景和模型
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # 设置背景颜色
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 渲染训练集
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        # 渲染测试集
        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    # 添加渲染特定参数
    parser.add_argument("--iteration", default=-1, type=int)  # 要渲染的迭代次数,-1 表示最新
    parser.add_argument("--skip_train", action="store_true")  # 跳过训练集渲染
    parser.add_argument("--skip_test", action="store_true")  # 跳过测试集渲染
    parser.add_argument("--quiet", action="store_true")  # 静默模式
    # 从模型目录加载已保存的参数
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # 初始化系统状态(随机数生成器)
    safe_state(args.quiet)

    # 执行渲染
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)
