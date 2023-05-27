import os
import oss2

class HandleOSS():
    def __init__(self, key_id='LTAI5tN8tqY8kUGp75Qa1r75', key_secret='h6xFTWYVJVyXKNFE4Tx3fbVqan6Ysx', bucket='obj-extend'):
        '''
        :param key_id:
        :param key_secret:
        :param bucket: bucket名字，例如：test
        '''
        self.auth = oss2.Auth(key_id, key_secret)
        self.link_url = 'https://oss-cn-nanjing.aliyuncs.com'
        if bucket:
            self.bucket = oss2.Bucket(self.auth, self.link_url, bucket)

    # 将一个文件夹下面的所有文件都上传
    def update_files(self, file_dir, oss_dir):
        '''
        :param file_dir: 要上传图片所在的文件夹，例如：/Users/guojun/Desktop/pic
        :param oss_dir: oss上的路径，要存在oss上哪个文件夹下面，例如：test/1212
        :return:
        '''
        for i in os.listdir(file_dir):
            dir = os.path.join(file_dir, i)
            for file in os.listdir(dir):
            # oss上传后的路径
                oss_path = f'{oss_dir}/{i}/{file}'
            # 本地文件路径
                file_path = f'{file_dir}/{i}/{file}'
            # 进行上传
                self.bucket.put_object_from_file(oss_path, file_path)

    def update_file(self, file_path, oss_dir):
        '''
        :param file_dir: 要上传图片所在的文件夹，例如：/Users/guojun/Desktop/pic
        :param oss_dir: oss上的路径，要存在oss上哪个文件夹下面，例如：test/1212
        :return:
        '''
        filename = file_path.split('\\')[-1]
                # oss上传后的路径
        oss_path = f'{oss_dir}/{filename}'
        # 本地文件路径
        file_path = f'{file_path}'
        # 进行上传
        self.bucket.put_object_from_file(oss_path, file_path)



    # 下载单个文件
    # def download_one_file(self, oss_path, save_dir):
    #     '''
    #     :param oss_path: 文件所在的oss地址，例如：test/test.text
    #     :param save_dir: 要保存在本地的文件目录，例如：/Users/guojun/Desktop/pic
    #     :return:
    #     '''
    #     file_name = oss_path.split('/')[-1]
    #     save_path = os.path.join(save_dir, file_name)
    #     result = self.bucket.get_object_to_file(oss_path, save_path)
    #     if result.status == 200:
    #         return 1

# if __name__ == '__main__':
#     # 填写oss的存储路径
#     project_name = 'test/'
#     h = HandleOSS()
#     h.update_file("D:/PycharmProjects/objAug_tool/server/extend/000000000139")



