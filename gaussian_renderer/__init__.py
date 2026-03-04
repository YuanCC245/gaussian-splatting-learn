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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    渲染 3D 高斯场景

    参数:
        viewpoint_camera: 视点相机
        pc: 3D 高斯点云模型
        pipe: 渲染管线参数
        bg_color: 背景颜色张量(必须在 GPU 上)
        scaling_modifier: 缩放修正因子
        separate_sh: 是否分离直流分量和其余球谐系数(用于 SparseAdam)
        override_color: 覆盖颜色(如果提供则不使用球谐函数)
        use_trained_exp: 是否使用训练的曝光参数

    返回:
        包含渲染图像、视空间点、可见性过滤器、半径和深度图的字典
    """

    # 创建零张量,用于让 PyTorch 返回 2D(屏幕空间)均值的梯度
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()  # 保留梯度用于密集化
    except:
        pass

    # 设置光栅化配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    # 创建光栅化器
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取高斯属性
    means3D = pc.get_xyz  # 3D 位置
    means2D = screenspace_points  # 2D 屏幕空间位置
    opacity = pc.get_opacity  # 不透明度

    # 如果提供了预计算的 3D 协方差矩阵,使用它；否则由光栅化器从缩放/旋转计算
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        # 在 Python 中预计算协方差
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # 由 CUDA 光栅化器计算协方差
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 如果提供了预计算的颜色,使用它；否则如果需要在 Python 中预计算球谐颜色,则执行；
    # 如果都不,则由光栅化器进行球谐到 RGB 的转换
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            # 在 Python 中预计算球谐颜色
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                # 分离直流分量和其余球谐系数(用于 SparseAdam 优化器)
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                # 合并所有球谐系数
                shs = pc.get_features
    else:
        # 使用覆盖颜色
        colors_precomp = override_color

    # 将可见的高斯光栅化到图像,获取它们的屏幕半径
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

    # 应用曝光补偿(仅训练时)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # 被视锥体裁剪或半径为 0 的高斯是不可见的,将从分裂标准中排除
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,  # 渲染的图像
        "viewspace_points": screenspace_points,  # 视空间点坐标
        "visibility_filter" : (radii > 0).nonzero(),  # 可见性过滤器
        "radii": radii,  # 屏幕空间半径
        "depth" : depth_image  # 深度图
        }

    return out
