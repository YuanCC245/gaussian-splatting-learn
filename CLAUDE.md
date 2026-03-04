# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

这是 Kerbl 等人(SIGGRAPH 2023)的 3D Gaussian Splatting 实时辐射场渲染的官方实现。该代码库从 SfM(运动恢复结构)输入训练 3D 高斯模型并实时渲染。

## 环境配置

```bash
# 克隆仓库(包含子模块)
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive

# 创建 conda 环境
conda env create --file environment.yml
conda activate gaussian_splatting
```

环境依赖：
- Python 3.7.13
- PyTorch 1.12.1 with CUDA 11.6
- Git 子模块：`diff-gaussian-rasterization`、`simple-knn`、`fused-ssim`、`SIBR_viewers`

## 核心命令

### 训练
```bash
# 基础训练
python train.py -s <COLMAP 或 NeRF 合成数据集路径>

# 使用评估划分训练
python train.py -s <路径> --eval

# 指定输出目录
python train.py -s <路径> -m <输出路径>

# 从检查点继续训练
python train.py -s <路径> --start_checkpoint <检查点路径>
```

### 渲染与评估
```bash
# 从训练好的模型生成渲染图像
python render.py -m <训练模型路径>

# 计算误差指标(SSIM、PSNR、LPIPS)
python metrics.py -m <训练模型路径>

# 完整评估流程(训练、渲染、指标计算)
python full_eval.py -m360 <mipnerf360文件夹> -tat <tanks_and_temples> -db <deep_blending>
```

### 处理自定义数据集
```bash
# 将图像转换为 COLMAP 格式
python convert.py -s <位置> [--resize]
```

## 代码架构

### 核心脚本
- **`train.py`**：主训练循环,包含高斯密集化/剪枝、损失计算(L1 + SSIM)、可选的深度正则化和曝光补偿
- **`render.py`**：从保存的检查点渲染训练好的模型
- **`metrics.py`**：使用 LPIPS、SSIM、PSNR 计算评估指标
- **`convert.py`**：将原始图像转换为训练用的 COLMAP 数据集格式

### 关键模块

**`scene/`**：场景表示和数据加载
- **`GaussianModel`**：存储所有高斯属性(xyz、features_dc、features_rest、scaling、rotation、opacity、exposure),处理密集化、剪枝和优化器设置
- **`Scene`**：管理训练/测试相机,加载/保存模型,处理数据集读取器
- **`dataset_readers.py`**：加载 COLMAP 和 Blender/NeRF 合成数据集
- **`cameras.py`**：支持 COLMAP/NeRF 参数的相机类

**`gaussian_renderer/`**：可微分光栅化
- 使用 `diff_gaussian_rasterization` CUDA 扩展
- 支持抗锯齿(来自 Mip Splatting 的 EWA 滤波器)
- 返回渲染图像、可见性过滤器、半径和深度

**`arguments/`**：参数管理
- **`ModelParams`**：数据集路径、分辨率、背景颜色、球谐阶数
- **`OptimizationParams`**：学习率、密集化设置、损失权重、优化器类型
- **`PipelineParams`**：渲染选项(convert_SHs_python、compute_cov3D_python、debug、antialiasing)

**`utils/`**：工具函数
- **`loss_utils.py`**：L1 损失、SSIM
- **`image_utils.py`**：PSNR 指标
- **`sh_utils.py`**：球谐函数
- **`graphics_utils.py`**：3D 变换、BasicPointCloud

### 训练流程

1. **初始化**：从 COLMAP/NeRF 数据加载场景,从点云初始化高斯
2. **训练循环**(默认 30k 次迭代)：
   - 渲染随机相机视角
   - 计算损失：`(1.0 - lambda_dssim) * L1 + lambda_dssim * (1.0 - SSIM)`
   - 可选：如果提供了深度图,添加深度 L1 损失
   - 反向传播
   - 每 1000 次迭代：增加球谐阶数
   - 高斯密集化/剪枝(迭代 500-15000)
   - 每 3000 次迭代重置不透明度
   - 优化器步进(Adam 或 SparseGaussianAdam)

### 可选功能

**深度正则化**(`-d <depths_dir>`)：使用单目深度图改善重建,特别是在无纹理区域

**曝光补偿**：为每张图像优化仿射变换以处理曝光变化
```
--exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --train_test_exp
```

**训练加速**(`--optimizer_type sparse_adam`)：需要切换到加速光栅化器分支
```bash
pip uninstall diff-gaussian-rasterization -y
cd submodules/diff-gaussian-rasterization
git checkout 3dgs_accel
pip install .
```

**抗锯齿**(`--antialiasing`)：启用 EWA 滤波器减少混叠伪影

### 模型输出结构

```
<model_path>/
├── point_cloud/
│   └── iteration_<N>/
│       └── point_cloud.ply        # 训练好的高斯模型
├── cameras.json                    # 相机参数
├── cfg_args                        # 训练配置
├── exposure.json                   # 曝光值(如果使用了 train_test_exp)
├── chkpnt<N>.pth                   # 检查点(如果指定)
└── train/                          # Tensorboard 日志(如果可用)
```

## 重要说明

- 宽度超过 1600px 的图像默认会自动缩放；使用 `-r 1` 强制使用完整分辨率
- VRAM 使用量随高斯数量增加；对于 VRAM 有限的情况,可降低 `--densify_grad_threshold` 或 `--densify_until_iter`
- 使用预训练模型评估时,需要通过 `-s <source_path>` 向 render.py 提供原始源数据
- SIBR 查看器(位于 `SIBR_viewers/`)是一个用于实时可视化的独立 C++ 项目
