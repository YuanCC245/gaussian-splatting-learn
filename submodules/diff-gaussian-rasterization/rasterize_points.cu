/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

/**
 * 创建 PyTorch 张量的调整大小函数
 *
 * 该函数返回一个 lambda，用于在 CUDA 核心中动态调整 PyTorch 张量的大小
 * 这对于在光栅化过程中分配不确定大小的缓冲区非常有用
 *
 * 参数:
 *   t: 要调整大小的 PyTorch 张量
 *
 * 返回:
 *   一个可调用对象，接受新大小 N 并返回调整后的张量数据指针
 */
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});  // 调整张量大小
		return reinterpret_cast<char*>(t.contiguous().data_ptr());  // 返回连续内存的指针
    };
    return lambda;
}

/**
 * 前向渲染：将 3D 高斯投影到 2D 图像
 *
 * 该函数是 Python 和 CUDA 之间的接口，负责：
 * 1. 验证输入张量的维度
 * 2. 分配输出张量和缓冲区
 * 3. 调用 CUDA 光栅化核心
 * 4. 返回渲染结果和中间数据（用于反向传播）
 *
 * 参数:
 *   background: (3,) 背景颜色
 *   means3D: (N, 3) 高斯中心的 3D 坐标
 *   colors: (N, 3) 预计算的颜色（如果提供，sh 将被忽略）
 *   opacity: (N, 1) 不透明度值
 *   scales: (N, 3) 缩放因子
 *   rotations: (N, 4) 旋转四元数
 *   scale_modifier: 全局缩放修正因子
 *   cov3D_precomp: (N, 6) 预计算的 3D 协方差矩阵（上三角部分）
 *   viewmatrix: (4, 4) 视图变换矩阵
 *   projmatrix: (4, 4) 投影矩阵
 *   tan_fovx: X 方向视野角正切值
 *   tan_fovy: Y 方向视野角正切值
 *   image_height: 输出图像高度
 *   image_width: 输出图像宽度
 *   sh: (N, M, 3) 球谐函数系数，M = (sh_degree + 1)^2
 *   degree: 球谐函数的最大阶数 (0-3)
 *   campos: (3,) 相机位置
 *   prefiltered: 是否使用预过滤（优化）
 *   antialiasing: 是否启用抗锯齿（EWA 滤波）
 *   debug: 是否启用调试模式
 *
 * 返回:
 *   std::tuple 包含:
 *     - rendered: 渲染的高斯数量
 *     - out_color: (3, H, W) 渲染的彩色图像
 *     - radii: (N,) 每个高斯在屏幕空间的半径
 *     - geomBuffer: 几何缓冲区（用于反向传播）
 *     - binningBuffer: 分块缓冲区（用于反向传播）
 *     - imgBuffer: 图像缓冲区（用于反向传播）
 *     - out_invdepth: (1, H, W) 逆深度图
 */
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug)
{
  // 验证输入张量的维度
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  // 获取输入参数
  const int P = means3D.size(0);  // 高斯数量
  const int H = image_height;      // 图像高度
  const int W = image_width;       // 图像宽度

  // 设置输出张量的数据类型选项
  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  // 创建输出张量：渲染的彩色图像 (3, H, W)
  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);

  // 创建逆深度图输出张量
  torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts);
  float* out_invdepthptr = nullptr;

  out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  out_invdepthptr = out_invdepth.data<float>();

  // 创建屏幕半径张量：每个高斯在屏幕空间的半径
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  // 创建用于反向传播的缓冲区（几何、分块、图像）
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));

  // 创建缓冲区调整大小的函数对象
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  // 如果有高斯点，执行渲染
  if(P != 0)
  {
	  // 获取球谐系数数量
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);  // 球谐系数数量 = (degree + 1)^2
      }

	  // 调用 CUDA 光栅化器的前向渲染函数
	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,           // 几何缓冲区调整函数
		binningFunc,        // 分块缓冲区调整函数
		imgFunc,            // 图像缓冲区调整函数
	    P, degree, M,       // 高斯数量、球谐阶数、球谐系数数量
		background.contiguous().data<float>(),      // 背景颜色指针
		W, H,               // 图像宽度和高度
		means3D.contiguous().data<float>(),         // 3D 位置指针
		sh.contiguous().data_ptr<float>(),           // 球谐系数指针
		colors.contiguous().data<float>(),           // 预计算颜色指针
		opacity.contiguous().data<float>(),          // 不透明度指针
		scales.contiguous().data_ptr<float>(),       // 缩放指针
		scale_modifier,                             // 缩放修正因子
		rotations.contiguous().data_ptr<float>(),    // 旋转指针
		cov3D_precomp.contiguous().data<float>(),    // 预计算协方差指针
		viewmatrix.contiguous().data<float>(),       // 视图矩阵指针
		projmatrix.contiguous().data<float>(),       // 投影矩阵指针
		campos.contiguous().data<float>(),           // 相机位置指针
		tan_fovx,                                   // X 方向视野角正切值
		tan_fovy,                                   // Y 方向视野角正切值
		prefiltered,                                // 预过滤标志
		out_color.contiguous().data<float>(),       // 输出颜色指针
		out_invdepthptr,                            // 输出逆深度指针
		antialiasing,                               // 抗锯齿标志
		radii.contiguous().data<int>(),             // 输出半径指针
		debug);                                     // 调试标志
  }
  // 返回渲染结果和中间缓冲区
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth);
}

/**
 * 反向传播：计算各参数的梯度
 *
 * 该函数使用前向传播保存的中间缓冲区，计算所有可学习参数的梯度
 *
 * 参数:
 *   background: (3,) 背景颜色
 *   means3D: (N, 3) 高斯中心的 3D 坐标
 *   radii: (N,) 每个高斯在屏幕空间的半径
 *   colors: (N, 3) 预计算的颜色
 *   opacities: (N, 1) 不透明度值
 *   scales: (N, 3) 缩放因子
 *   rotations: (N, 4) 旋转四元数
 *   scale_modifier: 全局缩放修正因子
 *   cov3D_precomp: (N, 6) 预计算的 3D 协方差矩阵
 *   viewmatrix: (4, 4) 视图变换矩阵
 *   projmatrix: (4, 4) 投影矩阵
 *   tan_fovx: X 方向视野角正切值
 *   tan_fovy: Y 方向视野角正切值
 *   dL_dout_color: (3, H, W) 输出颜色的梯度
 *   dL_dout_invdepth: (1, H, W) 输出深度的梯度
 *   sh: (N, M, 3) 球谐函数系数
 *   degree: 球谐函数的最大阶数
 *   campos: (3,) 相机位置
 *   geomBuffer: 几何缓冲区（从前向保存）
 *   R: 渲染的高斯数量
 *   binningBuffer: 分块缓冲区（从前向保存）
 *   imageBuffer: 图像缓冲区（从前向保存）
 *   antialiasing: 抗锯齿标志
 *   debug: 调试标志
 *
 * 返回:
 *   std::tuple 包含各参数的梯度：
 *     - dL_dmeans2D: (N, 3) 2D 屏幕空间位置的梯度
 *     - dL_dcolors: (N, 3) 颜色的梯度
 *     - dL_dopacity: (N, 1) 不透明度的梯度
 *     - dL_dmeans3D: (N, 3) 3D 位置的梯度
 *     - dL_dcov3D: (N, 6) 3D 协方差的梯度
 *     - dL_dsh: (N, M, 3) 球谐系数的梯度
 *     - dL_dscales: (N, 3) 缩放的梯度
 *     - dL_drotations: (N, 4) 旋转的梯度
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
  const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_invdepth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,
	const bool debug)
{
  // 获取输入参数
  const int P = means3D.size(0);  // 高斯数量
  const int H = dL_dout_color.size(1);  // 图像高度
  const int W = dL_dout_color.size(2);  // 图像宽度

  // 获取球谐系数数量
  int M = 0;
  if(sh.size(0) != 0)
  {
	M = sh.size(1);  // 球谐系数数量 = (degree + 1)^2
  }

  // 创建梯度张量
  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());

  // 处理逆深度梯度（如果存在）
  float* dL_dinvdepthsptr = nullptr;
  float* dL_dout_invdepthptr = nullptr;
  if(dL_dout_invdepth.size(0) != 0)
  {
	dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
	dL_dinvdepths = dL_dinvdepths.contiguous();
	dL_dinvdepthsptr = dL_dinvdepths.data<float>();
	dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
  }

  // 如果有高斯点，执行反向传播
  if(P != 0)
  {
	  // 调用 CUDA 光栅化器的反向传播函数
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),      // 背景颜色指针
	  W, H,                                     // 图像宽度和高度
	  means3D.contiguous().data<float>(),         // 3D 位置指针
	  sh.contiguous().data<float>(),               // 球谐系数指针
	  colors.contiguous().data<float>(),           // 颜色指针
	  opacities.contiguous().data<float>(),        // 不透明度指针
	  scales.data_ptr<float>(),                   // 缩放指针
	  scale_modifier,                             // 缩放修正因子
	  rotations.data_ptr<float>(),                // 旋转指针
	  cov3D_precomp.contiguous().data<float>(),    // 预计算协方差指针
	  viewmatrix.contiguous().data<float>(),       // 视图矩阵指针
	  projmatrix.contiguous().data<float>(),       // 投影矩阵指针
	  campos.contiguous().data<float>(),           // 相机位置指针
	  tan_fovx,                                  // X 方向视野角正切值
	  tan_fovy,                                  // Y 方向视野角正切值
	  radii.contiguous().data<int>(),             // 屏幕半径指针
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),   // 几何缓冲区指针
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()), // 分块缓冲区指针
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),  // 图像缓冲区指针
	  dL_dout_color.contiguous().data<float>(),   // 输出颜色梯度指针
	  dL_dout_invdepthptr,                       // 输出深度梯度指针
	  dL_dmeans2D.contiguous().data<float>(),      // 2D 位置梯度指针
	  dL_dconic.contiguous().data<float>(),        // 2D 协方差梯度指针
	  dL_dopacity.contiguous().data<float>(),      // 不透明度梯度指针
	  dL_dcolors.contiguous().data<float>(),       // 颜色梯度指针
	  dL_dinvdepthsptr,                          // 逆深度梯度指针
	  dL_dmeans3D.contiguous().data<float>(),      // 3D 位置梯度指针
	  dL_dcov3D.contiguous().data<float>(),        // 3D 协方差梯度指针
	  dL_dsh.contiguous().data<float>(),           // 球谐系数梯度指针
	  dL_dscales.contiguous().data<float>(),       // 缩放梯度指针
	  dL_drotations.contiguous().data<float>(),    // 旋转梯度指针
	  antialiasing,                               // 抗锯齿标志
	  debug);                                    // 调试标志
  }

  // 返回各参数的梯度
  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

/**
 * 标记在相机视锥体内的可见点
 *
 * 该函数通过视锥体裁剪（frustum culling）标记哪些高斯点在当前视角下可见
 * 可见的高斯点将被标记为 true，不可见的被标记为 false
 *
 * 参数:
 *   means3D: (N, 3) 高斯中心的 3D 坐标
 *   viewmatrix: (4, 4) 视图变换矩阵
 *   projmatrix: (4, 4) 投影矩阵
 *
 * 返回:
 *   present: (N,) 布尔张量，True 表示该高斯在相机视锥体内可见
 */
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{
  const int P = means3D.size(0);  // 高斯数量

  // 创建可见性标记张量，初始值全部为 false
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

  // 如果有高斯点，执行视锥体裁剪
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),    // 3D 位置指针
		viewmatrix.contiguous().data<float>(),  // 视图矩阵指针
		projmatrix.contiguous().data<float>(),  // 投影矩阵指针
		present.contiguous().data<bool>());    // 可见性输出指针
  }

  return present;
}
