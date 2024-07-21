# 从未来版本导入print函数以确保与Python 3兼容
from __future__ import print_function
# 导入操作系统相关功能，系统特定参数和文件操作工具
import os
import sys
# 用于查找符合特定规则的文件路径名
import glob
# 导入PIL库中的Image模块并重命名为PILImage
from PIL import Image as PILImage

# 导入自定义工具函数，用于将.mat文件转换为.png图像
from utils import mat2png_hariharan

def main():
    """
    主函数，处理命令行参数，检查输入输出目录是否存在，
    并调用相应函数进行.mat到.png的转换。
    """
    # 处理命令行参数，获取输入输出路径
    input_path, output_path = process_arguments(sys.argv) 
    
    # 检查输入输出路径是否都是有效目录
    if os.path.isdir(input_path) and os.path.isdir(output_path):
        # 查找输入目录下所有.mat文件，并进行转换
        mat_files = glob.glob(os.path.join(input_path, '*.mat'))
        convert_mat2png(mat_files, output_path)
    else:
        # 若路径无效，显示帮助信息
        help('输入或输出路径不存在!\n')

def process_arguments(argv):
    """
    解析命令行参数，返回输入和输出路径。
    
    参数:
    argv -- 命令行参数列表。
    
    返回:
    一个包含输入路径和输出路径的元组。
    """
    num_args = len(argv)
    input_path = None
    output_path = None 

    # 检查参数数量是否正确，分配输入输出路径值
    if num_args == 3:
        input_path  = argv[1]
        output_path = argv[2]
    else:
        # 参数数量错误时，显示帮助信息
        help()

    return input_path, output_path

def convert_mat2png(mat_files, output_path):
    """
    将.mat文件批量转换为.png图像并保存至指定目录。
    
    参数:
    mat_files -- .mat文件路径列表。
    output_path -- 转换后.png图像的保存目录。
    """
    if not mat_files:
        # 若无.mat文件，显示提示信息
        help('输入目录中没有Matlab文件可供转换!\n')

    # 遍历.mat文件，转换并保存为.png格式
    for mat_file in mat_files:
        numpy_img = mat2png_hariharan(mat_file)
        pil_img = PILImage.fromarray(numpy_img)
        # 修改文件名扩展名并保存图像
        pil_img.save(os.path.join(output_path, modify_image_name(mat_file, 'png')))

def modify_image_name(path, ext):
    """
    修改文件名的扩展名。
    
    参数:
    path -- 文件原始路径。
    ext -- 新的扩展名。
    
    返回:
    修改扩展名后的新文件名（不含路径）。
    """
    # 分离文件名和原扩展名，替换为新扩展名
    return os.path.splitext(os.path.basename(path))[0] + '.' + ext

def help(msg=''):
    """
    打印使用帮助信息至标准错误输出，并退出程序。
    
    参数:
    msg -- 可选的附加错误信息。
    """
    print(msg +
          '用法: python mat2png.py 输入路径 输出路径\n'
          '输入路径应包含待转换的Matlab文件。\n'
          '输出路径是转换后的Png文件保存的位置。'
          , file=sys.stderr)
    sys.exit()

# 当脚本直接运行时执行主函数
if __name__ == '__main__':
    main()
