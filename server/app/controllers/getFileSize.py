import os

def formatsize(bytes):
    try:
        bytes = float(bytes)  # 默认字节
        kb = bytes / 1024  # 换算KB
    except:
        print("字节格式有误")
        return "Error"

    if kb >= 1024:
        M = kb / 1024  # KB换成M
        if M >= 1024:
            G = M / 1024
            return "%.2fGB" % G
        else:
            return "%.2fMB" % M
    else:
        return "%.2fkb" % kb

# 获取文件大小
def Getfile(path):
    try:
        size = os.path.getsize(path)
        return formatsize(size)
    except:
        print("获取文件大小错误")

# 获取目录总大小
def Getdir(filepath):  # 定义函数
    sum = 0  # 初始化文件大小
    try:
        filename = os.walk(filepath)  # 获取文件夹目录
        for root, dirs, files in filename:  # 循环遍历文件夹目录下的文件
            for fle in files:
                filesdirs = os.path.join(root, fle)  # 必须要这一步,不然获取的文件没有找到路径.
                filesize = os.path.getsize(filesdirs)  # 统计循环出来的文件大小
                sum += filesize  # 所有文件加起来总和
        return formatsize(sum)
    except:
        print("获取文件夹大小错误")


def count_files(path):
    count = 0
    for root, dirs, files in os.walk(path):
        count += len(files)
    return count


# save_dir = os.path.join(os.getcwd(), 'upload')
#         save_path = os.path.join(save_dir, filename)
#         if not os.path.exists(save_path):
#             os.mkdir(save_path)
#             r = zipfile.is_zipfile(address)
#             if r:
#                 fz = zipfile.ZipFile(address, 'r')
#                 for file in fz.namelist():
#                     fz.extract(file, save_path)
#             else:
#                 print('This is not zip')