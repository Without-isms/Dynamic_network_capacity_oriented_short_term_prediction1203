import os
import math
from datetime import datetime

# 文件夹路径
folder_path = f"..{os.path.sep}data{os.path.sep}suzhou{os.path.sep}OD{os.path.sep}"
folder_path = f"..{os.path.sep}data{os.path.sep}suzhou{os.path.sep}"
# 定义文件大小的转换函数
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))  # 使用 math.floor 和 math.log
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


# 获取文件信息
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # 获取文件大小
    file_size = os.path.getsize(file_path)
    readable_size = convert_size(file_size)

    # 获取文件创建时间
    creation_time = os.path.getctime(file_path)
    formatted_time = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')

    # 获取文件类型
    file_type = "文件" if os.path.isfile(file_path) else "文件夹"

    # 打印文件信息
    print(f"文件名: {filename}")
    print(f"创建时间: {formatted_time}")
    print(f"文件类型: {file_type}")
    print(f"文件大小: {readable_size}")
    print("-" * 40)