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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    """
    训练 3D 高斯模型的主函数

    参数:
        dataset: 数据集参数(ModelParams)
        opt: 优化参数(OptimizationParams)
        pipe: 渲染管线参数(PipelineParams)
        testing_iterations: 进行评估的迭代次数列表
        saving_iterations: 保存模型的迭代次数列表
        checkpoint_iterations: 保存检查点的迭代次数列表
        checkpoint: 要恢复的检查点路径
        debug_from: 从哪次迭代开始启用调试模式
    """

    # 检查 sparse_adam 优化器是否可用
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    # 初始化起始迭代
    first_iter = 0
    # 准备输出目录和日志记录器
    tb_writer = prepare_output_and_logger(dataset)
    # 创建高斯模型,指定球谐函数阶数和优化器类型
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    # 创建场景,加载数据集
    scene = Scene(dataset, gaussians)
    # 设置优化器
    gaussians.training_setup(opt)
    # 如果提供了检查点,从检查点恢复训练
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 设置背景颜色(白色或黑色)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 用于测量每次迭代时间的 CUDA 事件
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 是否使用 sparse_adam 优化器
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    # 深度损失的权重函数(指数衰减)
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # 获取训练相机
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    # 用于日志记录的指数移动平均损失
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # 创建进度条
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # 主训练循环
    for iteration in range(first_iter, opt.iterations + 1):
        # 尝试连接网络 GUI
        if network_gui.conn == None:
            network_gui.try_connect()
        # 处理 GUI 交互
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                # 从 GUI 接收相机和控制参数
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # 渲染自定义相机视角
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    # 将渲染结果转换为字节发送给 GUI
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # 记录迭代开始时间
        iter_start.record()

        # 更新学习率
        gaussians.update_learning_rate(iteration)

        # 每 1000 次迭代增加球谐函数阶数(最高到 max_sh_degree)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择一个相机视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # 渲染
        # 如果到达调试起始迭代,启用调试模式
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 使用随机背景或固定背景
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 渲染当前视角
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 应用 alpha 掩码(如果存在)
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # 计算损失
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        # 使用融合 SSIM 或普通 SSIM
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        # 总损失 = (1 - λ) * L1_loss + λ * (1 - SSIM)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # 深度正则化
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            # 计算深度损失
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # 反向传播
        loss.backward()

        # 记录迭代结束时间
        iter_end.record()

        with torch.no_grad():
            # 更新进度条(使用指数移动平均)
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录日志和保存
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            # 保存高斯模型
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 密集化与剪枝
            if iteration < opt.densify_until_iter:
                # 记录图像空间中的最大半径,用于剪枝
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 添加密集化统计信息
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 执行密集化和剪枝
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                # 定期重置不透明度
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步进
            if iteration < opt.iterations:
                # 更新曝光优化器
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                # 更新高斯参数优化器
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # 保存检查点
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    """
    准备输出目录和 TensorBoard 日志记录器

    参数:
        args: 包含模型路径等配置的参数

    返回:
        tb_writer: TensorBoard 写入器(如果可用)
    """
    # 如果没有指定模型路径,生成唯一路径
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # 创建输出文件夹
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    # 保存配置参数
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建 TensorBoard 写入器
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    """
    生成训练报告,记录指标到 TensorBoard 并在测试集上评估

    参数:
        tb_writer: TensorBoard 写入器
        iteration: 当前迭代次数
        Ll1: L1 损失
        loss: 总损失
        l1_loss: L1 损失函数
        elapsed: 本次迭代耗时
        testing_iterations: 进行评估的迭代次数列表
        scene: 场景对象
        renderFunc: 渲染函数
        renderArgs: 渲染函数参数
        train_test_exp: 是否使用训练/测试曝光
    """
    # 记录训练损失到 TensorBoard
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 在指定的测试迭代次数评估测试集和训练集样本
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 配置验证集(测试集和训练集样本)
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                # 渲染并计算指标
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # 如果使用曝光补偿,只使用右半部分图像
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    # 记录图像到 TensorBoard(仅前5个)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    # 累积 L1 和 PSNR
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                # 计算平均指标
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # 记录场景统计信息
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # 添加其他训练参数
    parser.add_argument('--ip', type=str, default="127.0.0.1")  # GUI 服务器 IP
    parser.add_argument('--port', type=int, default=6009)  # GUI 服务器端口
    parser.add_argument('--debug_from', type=int, default=-1)  # 从哪次迭代开始调试
    parser.add_argument('--detect_anomaly', action='store_true', default=False)  # 检测梯度异常
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])  # 测试迭代点
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])  # 保存迭代点
    parser.add_argument("--quiet", action="store_true")  # 静默模式
    parser.add_argument('--disable_viewer', action='store_true', default=False)  # 禁用查看器
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # 检查点迭代点
    parser.add_argument("--start_checkpoint", type=str, default = None)  # 起始检查点路径
    args = parser.parse_args(sys.argv[1:])
    # 确保在最终迭代时保存模型
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # 初始化系统状态(随机数生成器)
    safe_state(args.quiet)

    # 启动 GUI 服务器,配置并运行训练
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # 训练完成
    print("\nTraining complete.")
