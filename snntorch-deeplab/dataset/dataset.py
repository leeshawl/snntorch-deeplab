from torchvision.datasets import VOCSegmentation
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple, List
import torch
import numpy as np
import torchvision

# 定义一个PASCAL VOC数据集专用的颜色映射列表，用于将颜色编码为标签
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

def voc_rand_crop(feature, label, height, width):
    """
    随机裁剪特征图和标签图，使其大小变为指定的高度和宽度。

    参数:
    feature: 输入的特征图。
    label: 输入的标签图。
    height: 裁剪后输出图的高度。
    width: 裁剪后输出图的宽度。

    返回:
    裁剪后的特征图和标签图。
    """
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

def voc_colormap2label():
    """
    将颜色映射列表转换为一个长向量，用于快速将颜色编码为标签。

    返回:
    一个Tensor，包含颜色到标签的映射。
    """
    colormaplabel = torch.zeros(256 ** 3, dtype=torch.long)
    colormaplabel[:] = 255
    for i, colormap in enumerate(VOC_COLORMAP):
        idx = (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]
        colormaplabel[idx] = i
    return colormaplabel

def voc_label_indices(label, colormaplabel):
    """
    根据颜色映射将标签图转换为对应的类别索引。

    参数:
    label: 输入的标签图。
    colormaplabel: 颜色到标签的映射向量。

    返回:
    一个Tensor，包含每个像素的类别索引。
    """
    label = label
    idxs = [(label[:, :, 0] * 256 + label[:, :, 1]) * 256 + label[:, :, 2]]
    return colormaplabel[idxs]

class vocDataset(VOCSegmentation):
    """
    PASCAL VOC数据集的子类，用于加载和处理分割任务的数据。

    参数:
    root: 数据集根目录。
    year: 数据集的年份。
    image_set: 指定的数据集子集，如"train"、"val"等。
    download: 是否自动下载数据集。
    transform: 输入图像的转换。
    target_transform: 输入目标（标签图）的转换。
    transforms: 输入图像和目标的转换。
    """
    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    def __init__(self, root: str, year: str = "2012", image_set: str = "train", download: bool = False, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None):
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)
        # 初始化颜色到标签的映射
        self.colormap2label = voc_colormap2label()
        self.train = image_set  # 记录数据集子集是训练集还是其他

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        根据索引获取数据集中的一个样本（图像和对应的标签图）。

        参数:
        index: 样本的索引。

        返回:
        一个元组，包含图像和经过处理的标签。
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index]).convert("RGB")
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        target = np.array(target, dtype=np.int32)
        return img, voc_label_indices(target, self.colormap2label)
