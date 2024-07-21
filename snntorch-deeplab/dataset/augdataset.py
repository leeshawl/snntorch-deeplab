from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple, List
import torch
import numpy as np
from torch.utils.data import Dataset
import os

class augvocDataset(Dataset):
    """
    自定义数据集类，用于加载和处理augvoc格式的数据集。

    参数:
    - img_dir: 图像数据所在的目录。
    - im_set: 数据集类型，默认为"train"，可选"train"或"val"。
    - transform: 图像转换的函数或变换列表，可选。
    - target_transform: 目标图像转换的函数或变换列表，可选。
    """

    def __init__(self, img_dir, im_set="train", transform=None, target_transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform

        # 定义目标和图像的根目录
        target_root = os.path.join(img_dir, "aug_cls")
        img_root = os.path.join(img_dir, "train_aug")

        # 根据数据集类型加载对应的文件列表
        if im_set == "train":
            txt_path = os.path.join(img_dir, "train.txt")
            with open(txt_path, "r") as f:
                file_names = [x.strip() for x in f.readlines()]
            # 构建训练集图像和目标的路径列表
            self.images = [os.path.join(img_root, x.split(' ')[0] + ".jpg") for x in file_names]
            self.targets = [os.path.join(target_root, x.split(' ')[0] + ".png") for x in file_names]

        else:
            txt_path = os.path.join(img_dir, "val.txt")
            with open(txt_path, "r") as f:
                file_names = [x.strip() for x in f.readlines()]
            # 构建验证集图像和目标的路径列表
            self.images = [os.path.join(img_root, x + ".jpg") for x in file_names]
            self.targets = [os.path.join(target_root, x + ".png") for x in file_names]

        # 确保图像和目标的路径数量相等
        assert len(self.images) == len(self.targets)

    def __getitem__(self, index):
        """
        根据提供的索引获取数据集中的图像和标签。

        这个方法允许通过索引访问数据集中的每个样本，是实现数据集迭代器的关键部分。

        参数:
        index (int): 数据集中要访问的图像和标签对的索引。

        返回:
        tuple: 包含处理后的图像和标签的元组。
        """
        # 根据给定的索引获取图像和对应的标注。
        img_path = self.images[index]
        target_path = self.targets[index]
        print(f"尝试打开图像文件: {img_path}")  # 调试语句
        print(f"尝试打开目标文件: {target_path}")  # 调试语句

        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件未找到: {img_path}")

        # 打开并转换图像文件为RGB格式
        img = Image.open(img_path).convert("RGB")

        # 如果目标文件路径为None，返回None
        if target_path is None:
            target = None
        else:
            # 检查目标文件是否存在
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"目标文件未找到: {target_path}")
            # 打开并转换标签文件为P格式（用于分类）
            target = Image.open(target_path).convert("P")

        # 如果设置了图像转换函数，则应用到图像
        if self.transform is not None:
            img = self.transform(img)
        # 如果设置了标签转换函数，则应用到标签
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)

        # 将标签转换为numpy数组格式，以便进一步处理
        if target is not None:
            target = np.array(target)

        return img, target

    def __len__(self):
        """
        返回数据集的大小。

        返回:
        - len(self.images): 图像路径列表的长度。
        """
        return len(self.images)

if __name__ == "__main__":
    # 导入图像处理变换模块
    from torchvision import transforms

    # 定义批量处理的大小
    batch_size = 16

    # 定义图像的平均值和标准差，用于归一化
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # 构建图像变换序列，包括调整大小、转换为张量和归一化
    img_transform = transforms.Compose([
        transforms.Resize([64,64]),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])

    # 为训练集构建相同的图像变换序列
    train_img_transform = transforms.Compose([
        transforms.Resize([64,64]),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])

    # 构建目标（标签）的变换序列，仅调整大小
    tar_transform = transforms.Compose([transforms.Resize([64,64])])

    # 为测试集构建目标（标签）的变换序列，同样仅调整大小
    test_tar_transform = transforms.Compose([transforms.Resize([64,64])])

    # 加载训练数据集，应用图像和目标的变换
    dataset = augvocDataset("C:/Users/93074/Documents/VOC2012/VOC2012/ImageSets/Segmentation", transform=train_img_transform, target_transform=tar_transform)

    # 加载测试数据集，应用相同的图像和目标的变换
    test = augvocDataset("C:/Users/93074/Documents/VOC2012/VOC2012/ImageSets/Segmentation", "val", transform=train_img_transform, target_transform=tar_transform)

    # 遍历训练数据集，打印图像和标签的形状，用于调试
    for i, p in iter(dataset):
        print(i.shape, p.shape)
        break

    # 遍历测试数据集，打印图像和标签的形状，用于调试
    for i, p in iter(test):
        print(i.shape, p.shape)
        break