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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:
    """
    场景类：管理 3D 高斯模型的训练/渲染场景

    负责加载和管理：
    - 训练和测试相机
    - 高斯模型(点云)
    - 场景元数据(相机参数、曝光值等)
    """

    gaussians : GaussianModel  # 高斯模型对象

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        初始化场景

        参数:
            args: 模型参数(数据集路径、分辨率、背景色等)
            gaussians: 高斯模型对象
            load_iteration: 要加载的模型迭代次数,-1 表示加载最新的
            shuffle: 是否打乱相机顺序
            resolution_scales: 支持的分辨率缩放比例列表
        """
        self.model_path = args.model_path  # 模型保存路径
        self.loaded_iter = None  # 加载的迭代次数
        self.gaussians = gaussians  # 高斯模型引用

        # 如果指定了要加载的迭代次数
        if load_iteration:
            if load_iteration == -1:
                # -1 表示查找并加载最新的迭代
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # 初始化训练和测试相机字典(按分辨率缩放)
        self.train_cameras = {}
        self.test_cameras = {}

        # 根据数据集类型加载场景信息
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # COLMAP 数据集格式
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            # Blender/NeRF 合成数据集格式
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # 如果不是加载已有模型,则初始化新模型
        if not self.loaded_iter:
            # 复制初始点云到模型目录
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            # 保存相机信息为 JSON
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 打乱相机顺序(用于多分辨率一致性的随机打乱)
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # 多分辨率一致随机打乱
            random.shuffle(scene_info.test_cameras)  # 多分辨率一致随机打乱

        # 获取场景范围(用于密集化等操作)
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 为每个分辨率缩放创建相机对象
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        # 加载或创建高斯模型
        if self.loaded_iter:
            # 从保存的 PLY 文件加载已有模型
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            # 从点云初始化新的高斯模型
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        """
        保存当前迭代的高斯模型

        参数:
            iteration: 当前迭代次数
        """
        # 创建保存路径
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        # 保存高斯模型为 PLY 文件
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # 保存曝光参数
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        """
        获取训练相机列表

        参数:
            scale: 分辨率缩放比例

        返回:
            训练相机列表
        """
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        """
        获取测试相机列表

        参数:
            scale: 分辨率缩放比例

        返回:
            测试相机列表
        """
        return self.test_cameras[scale]
