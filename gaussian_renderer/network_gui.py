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
import traceback
import socket
import json
from scene.cameras import MiniCam

# 默认网络 GUI 配置
host = "127.0.0.1"  # 默认 IP 地址
port = 6009  # 默认端口

# 全局连接变量
conn = None  # 连接对象
addr = None  # 连接地址

# 创建 TCP 套接字
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    """
    初始化网络 GUI 服务器

    参数:
        wish_host: 期望的监听地址
        wish_port: 期望的监听端口
    """
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))  # 绑定地址和端口
    listener.listen()  # 开始监听
    listener.settimeout(0)  # 设置为非阻塞模式

def try_connect():
    """
    尝试接受客户端连接

    在每次训练迭代时调用,非阻塞方式
    """
    global conn, addr, listener
    try:
        conn, addr = listener.accept()  # 接受连接
        print(f"\nConnected by {addr}")
        conn.settimeout(None)  # 设置为阻塞模式
    except Exception as inst:
        pass  # 无连接可接受时忽略

def read():
    """
    从连接读取消息

    返回:
        解码后的 JSON 消息
    """
    global conn
    # 读取 4 字节的消息长度(小端序)
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    # 读取消息内容
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify):
    """
    发送消息到客户端

    参数:
        message_bytes: 图像字节流(可为 None)
        verify: 验证字符串(通常是数据集路径)
    """
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)  # 发送图像数据
    # 发送验证字符串长度和内容
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))

def receive():
    """
    接收来自 GUI 的消息并解析相机参数

    返回:
        custom_cam: 自定义相机对象(如果分辨率非零)
        do_training: 是否继续训练
        do_shs_python: 是否在 Python 中计算球谐函数
        do_rot_scale_python: 是否在 Python 中计算旋转和缩放
        keep_alive: 是否保持连接
        scaling_modifier: 缩放修正因子
    """
    message = read()

    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            # 解析相机参数
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            # 解析视图矩阵(4x4)
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            # 转换 Y 和 Z 轴(OpenGL/OpenGL 坐标系差异)
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            # 解析投影矩阵(4x4)
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            # 创建微型相机对象
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        # 分辨率为 0 表示无渲染请求
        return None, None, None, None, None, None
