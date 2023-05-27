import zipfile, os

def zipFile(srmdir_all_folder, zip_file_path):
    zip_file = zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED)  # 创建空的 zip文件
    for dirpath, dirnames, filenames in os.walk(srmdir_all_folder):
        fpath = dirpath.replace(srmdir_all_folder, '')  # 获取 相对文件夹的路径
        fpath = fpath and fpath + os.sep or os.sep  # 添加 '/'
        for filename in filenames:
            zip_file.write(os.path.join(dirpath, filename), fpath + filename)
    zip_file.close()