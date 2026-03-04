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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C  # 导入编译的 C++/CUDA 扩展模块

def cpu_deep_copy_tuple(input_tuple):
    """
    将元组中的张量深拷贝到 CPU

    参数:
        input_tuple: 包含张量和其他对象的元组

    返回:
        张量在 CPU 上的深拷贝元组
    """
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    """
    调用可微分的高斯光栅化函数

    参数:
        means3D: (N, 3) 高斯中心在 3D 空间的坐标
        means2D: (N, 3) 高斯中心在 2D 屏幕空间的坐标(用于梯度计算)
        sh: (N, M, 3) 球谐函数系数,M = (sh_degree + 1)^2
        colors_precomp: (N, 3) 预计算的颜色(如果提供,sh 将被忽略)
        opacities: (N, 1) 不透明度值
        scales: (N, 3) 缩放因子
        rotations: (N, 4) 旋转四元数
        cov3Ds_precomp: (N, 6) 预计算的 3D 协方差矩阵(上三角部分)
        raster_settings: GaussianRasterizationSettings 渲染配置

    返回:
        (color, radii, invdepths) 渲染图像、屏幕半径、逆深度图
    """
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    """
    可微分的高斯光栅化函数

    继承自 torch.autograd.Function,实现自定义的前向和反向传播
    """

    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        """
        前向传播：将 3D 高斯投影到 2D 图像

        参数:
            ctx: 上下文对象,用于保存反向传播所需的信息
            (其他参数同 rasterize_gaussians)

        返回:
            color: (3, H, W) 渲染的彩色图像
            radii: (N,) 每个高斯在屏幕空间的半径
            invdepths: (1, H, W) 逆深度图
        """

        # 按照底层 C++ 库期望的格式重组参数
        args = (
            raster_settings.bg,  # 背景颜色
            means3D,             # 3D 位置
            colors_precomp,      # 预计算颜色
            opacities,           # 不透明度
            scales,              # 缩放
            rotations,           # 旋转
            raster_settings.scale_modifier,  # 缩放修正因子
            cov3Ds_precomp,      # 预计算协方差
            raster_settings.viewmatrix,      # 视图矩阵 (4x4)
            raster_settings.projmatrix,      # 投影矩阵 (4x4)
            raster_settings.tanfovx,         # 视野角正切值 X
            raster_settings.tanfovy,         # 视野角正切值 Y
            raster_settings.image_height,    # 图像高度
            raster_settings.image_width,     # 图像宽度
            sh,                  # 球谐系数
            raster_settings.sh_degree,       # 球谐阶数
            raster_settings.campos,          # 相机位置 (3,)
            raster_settings.prefiltered,     # 是否预过滤
            raster_settings.antialiasing,    # 是否启用抗锯齿
            raster_settings.debug            # 是否启用调试模式
        )

        # 调用 C++/CUDA 光栅化器执行前向渲染
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        # 保存反向传播所需的张量
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):
        """
        反向传播：计算各参数的梯度

        参数:
            ctx: 上下文对象,包含前向传播保存的信息
            grad_out_color: (3, H, W) 输出颜色的梯度
            _: None(radii 不需要梯度)
            grad_out_depth: (1, H, W) 输出深度的梯度

        返回:
            各输入参数的梯度元组
        """

        # 从上下文恢复必要的值
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # 按 C++ 方法期望的格式重组参数
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                opacities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,       # 输入颜色梯度
                grad_out_depth,       # 输入深度梯度
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,           # 几何缓冲区(从前向保存)
                num_rendered,         # 渲染的高斯数量
                binningBuffer,        # 分块缓冲区
                imgBuffer,            # 图像缓冲区
                raster_settings.antialiasing,
                raster_settings.debug)

        # 调用 C++ 反向方法计算各参数的梯度
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        # 按输入参数顺序返回梯度(最后一个参数 raster_settings 返回 None)
        grads = (
            grad_means3D,           # means3D 的梯度
            grad_means2D,           # means2D 的梯度
            grad_sh,                # 球谐系数的梯度
            grad_colors_precomp,    # 预计算颜色的梯度
            grad_opacities,         # 不透明度的梯度
            grad_scales,            # 缩放的梯度
            grad_rotations,         # 旋转的梯度
            grad_cov3Ds_precomp,    # 协方差的梯度
            None,                   # raster_settings 不需要梯度
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    """
    高斯光栅化配置参数

    使用 NamedTuple 定义的不可变配置类
    """
    image_height: int          # 输出图像高度
    image_width: int           # 输出图像宽度
    tanfovx : float            # X 方向视野角正切值
    tanfovy : float            # Y 方向视野角正切值
    bg : torch.Tensor          # 背景颜色 (3,)
    scale_modifier : float     # 全局缩放修正因子
    viewmatrix : torch.Tensor  # 视图变换矩阵 (4,4)
    projmatrix : torch.Tensor  # 投影矩阵 (4,4)
    sh_degree : int            # 球谐函数阶数 (0-3)
    campos : torch.Tensor      # 相机位置 (3,)
    prefiltered : bool         # 是否使用预过滤(优化)
    debug : bool               # 是否启用调试模式
    antialiasing : bool        # 是否启用抗锯齿(EWA 滤波)

class GaussianRasterizer(nn.Module):
    """
    高斯光栅化器

    PyTorch nn.Module 包装类,提供友好的渲染接口
    """

    def __init__(self, raster_settings):
        """
        初始化光栅化器

        参数:
            raster_settings: GaussianRasterizationSettings 渲染配置
        """
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        """
        标记在相机视锥体内的可见点

        基于视锥体裁剪(frustum culling),用布尔标记可见点

        参数:
            positions: (N, 3) 3D 点的位置

        返回:
            visible: (N,) 布尔张量,True 表示点可见
        """
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        """
        前向渲染

        参数:
            means3D: (N, 3) 高斯中心的 3D 坐标
            means2D: (N, 3) 高斯中心的 2D 屏幕坐标
            opacities: (N, 1) 不透明度
            shs: (N, M, 3) 球谐函数系数(与 colors_precomp 二选一)
            colors_precomp: (N, 3) 预计算的颜色(与 shs 二选一)
            scales: (N, 3) 缩放因子(与 cov3D_precomp 二选一)
            rotations: (N, 4) 旋转四元数(与 cov3D_precomp 二选一)
            cov3D_precomp: (N, 6) 预计算的 3D 协方差(与 scales/rotations 二选一)

        返回:
            (color, radii, invdepths) 渲染结果

        异常:
            Exception: 当参数组合不正确时抛出
        """

        raster_settings = self.raster_settings

        # 验证颜色参数：必须提供 shs 或 colors_precomp 之一,不能同时提供
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        # 验证几何参数：必须提供 scales/rotations 或 cov3D_precomp 之一,不能同时提供
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        # 将未提供的参数设为空张量
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # 调用 C++/CUDA 光栅化函数执行渲染
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )
