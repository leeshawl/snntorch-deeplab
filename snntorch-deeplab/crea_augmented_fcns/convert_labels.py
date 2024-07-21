from __future__ import print_function
import os
import sys
from skimage.io import imread, imsave
from utils import convert_from_color_segmentation
def main():
    """
    主函数，程序的入口点。
    它负责解析命令行参数，读取图像文件列表，并将指定的图像从一种格式转换为另一种格式。
    """
    ##
    ext = '.png'  ## 图像文件的扩展名
    ##

    # 解析命令行参数，获取原始图像路径、图像列表文件名和转换后图像的保存路径
    path, txt_file, path_converted = process_arguments(sys.argv)

    # 如果转换后的图像路径不存在，则创建该目录
    # Create dir for converted labels
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    # 逐行读取图像列表文件，处理每张图像
    with open(txt_file, 'rb') as f:
        for img_name in f:
            img_base_name = str(img_name.strip(), encoding="utf8")
            img_name = os.path.join(path, img_base_name) + ext
            img = imread(img_name)

            # 如果图像为彩色，转换图像格式；否则输出错误信息并退出程序
            if (len(img.shape) > 2):
                img = convert_from_color_segmentation(img)
                imsave(os.path.join(path_converted, img_base_name) + ext, img)
            else:
                print(img_name + " is not composed of three dimensions, therefore " 
                      "shouldn't be processed by this script.\n"
                      "Exiting." , file=sys.stderr)
                exit()

def process_arguments(argv):
    """
    解析命令行参数，并返回原始图像路径、图像列表文件名和转换后图像的保存路径。

    参数:
    argv -- 命令行参数列表

    返回:
    path -- 原始图像路径
    list_file -- 图像列表文件名
    new_path -- 转换后图像的保存路径
    """
    if len(argv) != 4:
        help()

    path = argv[1]
    list_file = argv[2]
    new_path = argv[3]

    return path, list_file, new_path

def help():
    """
    输出帮助信息，说明如何使用本脚本，并退出程序。
    """
    print('Usage: python convert_labels.py PATH LIST_FILE NEW_PATH\n'
          'PATH points to directory with segmentation image labels.\n'
          'LIST_FILE denotes text file containing names of images in PATH.\n'
          'Names do not include extension of images.\n'
          'NEW_PATH points to directory where converted labels will be stored.'
          , file=sys.stderr)
    exit()

if __name__ == '__main__':
    main()
