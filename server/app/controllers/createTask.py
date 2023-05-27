import os, zipfile
import cv2, math, random
import numpy as np
from app.controllers.getFileSize import count_files


class TaskController:

    @classmethod
    def handle_task(cls, filename, taskname, address, methodList, multiple, timestamp):
        dataset_dir = os.path.join(os.getcwd(), 'upload')
        dataset_path = os.path.join(dataset_dir, filename)
        ext_dir = os.path.join(os.getcwd(), 'extend')
        ext_path = os.path.join(ext_dir, timestamp)
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
            r = zipfile.is_zipfile(address)
            if r:
                fz = zipfile.ZipFile(address, 'r')
                for file in fz.namelist():
                    fz.extract(file, dataset_path)
            else:
                print('This is not zip')
        ext_num = count_files(os.path.join(dataset_path, 'images')) * multiple

        obj_num = ext_num

        obj_handler = objController(filename, taskname, dataset_path, ext_path, obj_num)
        # obj_handler.insert_obj()
        obj_handler.remove_obj()





class objController:
    def __init__(self, filename, taskname, dataset_path, ext_path, obj_num, insertAtr=[1,1,1,1], removeAtr=[1,1], replaceAtr=[1,1,1,1]):
        self.insertAtr = insertAtr
        self.removeAtr = removeAtr
        self.replaceAtr = replaceAtr
        self.filename = filename
        self.taskname = taskname
        self.dataset_path = dataset_path
        self.ext_path = ext_path
        self.obj_num = int(obj_num)
        self.insert_num = obj_num // 3
        self.remove_num = obj_num // 3
        self.replace_num = obj_num // 3

    def save_dir(self):
        save_img_dir = os.path.join(self.ext_path, 'images')
        save_label_dir = os.path.join(self.ext_path, 'labels')
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_label_dir, exist_ok=True)
        return save_img_dir, save_label_dir

    def insert_obj(self):
        mask_dir = os.path.join(os.getcwd(), 'obj')
        img_dir = os.path.join(self.save_path, 'images')
        label_dir = os.path.join(self.save_path, 'labels')
        save_img_dir, save_label_dir = self.dataset_dir()
        print(save_img_dir)

        self.gen_mask(img_dir, label_dir, mask_dir, save_img_dir, save_label_dir)

    def add_alpha_channel(self, img):
        """ 为jpg图像添加alpha通道 """

        b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道

        img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
        return img_new

    def add_mask(self, source_path, mask_path, mask_class, resize_rate=0.05, random_pos=True, pos_h=None, pos_w=None):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        source_img = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)

        h, w = mask_img.shape[0], mask_img.shape[1]
        H, W = source_img.shape[0], source_img.shape[1]
        k = w / h

        if source_img.ndim == 2:  # 灰度图片先转成三通道的图片
            source_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2RGB)

        # 判断jpg图像是否已经为4通道
        if source_img.shape[2] == 3:
            source_img = self.add_alpha_channel(source_img)

        # 面积比 (w * (w/k))/ (W*H) <= resize_rate
        resize_w = int(math.sqrt(resize_rate * H * W * k)) if int(math.sqrt(resize_rate * H * W * k)) >= 1 else 1
        resize_h = int(resize_w / k) if int(resize_w / k) >= 1 else 1  # 不能变成 0了，至少要是1
        resized_mask_img = cv2.resize(mask_img, dsize=(resize_w, resize_h))
        if random_pos:  # pos是mask的左上角坐标，只要长宽不超出原图的边界即可
            pos_h = random.randint(0, H - resize_h)
            pos_w = random.randint(0, W - resize_w)
        else:  # 固定 mask 的位置，需要检查是否超出边界
            assert pos_w is not None
            assert pos_h is not None
            resize_w = W - pos_w if (pos_w + resize_w > W) else resize_w
            resize_h = H - pos_h if (pos_h + resize_h > H) else resize_h

        # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
        alpha_png = resized_mask_img[0:resize_h, 0:resize_w, 3] / 255.0
        alpha_jpg = 1 - alpha_png
        # 开始叠加
        for c in range(0, 3):
            source_img[pos_h:(pos_h + resize_h), pos_w:(pos_w + resize_w), c] = (
                    (alpha_jpg * source_img[pos_h:(pos_h + resize_h), pos_w:(pos_w + resize_w), c]) + (
                        alpha_png * resized_mask_img[0:resize_h, 0:resize_w, c]))
        # print("resize mask w:", resize_w, "resize mask h:", resize_h, "\npos w:", pos_w, "pos h:", pos_h)
        # print("source w:", W, "source h:", H)
        yolo_label_info = self.yolo_label(pos_w, pos_h, pos_w + resize_w, pos_h + resize_h, H, W, mask_class)
        return source_img, yolo_label_info

    def yolo_label(self, x1, y1, x2, y2, H, W, mask_class):
        new_x = (x1 + x2) / (2 * W)
        new_y = (y1 + y2) / (2 * H)

        new_w = (x2 - x1) / W
        new_h = (y2 - y1) / H

        # yolo_label_info = str(mask_class) + ' ' + new_x + ' ' + new_y + ' ' + new_w + ' ' + new_h + '\n'
        yolo_label_info = '{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(str(mask_class), new_x, new_y, new_w, new_h)
        return yolo_label_info

    def gen_mask(self, img_dir, label_dir, mask_dir, save_img_dir, save_label_dir):# insertion
        num = 0
        while num < self.insert_num:
            num += 1
            _files = os.listdir(img_dir)
            number = random.randint(0, len(_files) - 1)
            img_file = _files[number]
            fname = img_file.split('.jpg')[0]
            if os.path.exists(os.path.join(label_dir, fname + '.txt')):  # 找到匹配的 jpg 和 txt
                source_path = os.path.join(img_dir, img_file)
                txt_path = os.path.join(label_dir, fname + '.txt')
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                random_mask = random.sample(os.listdir(mask_dir), 1)
                mask_path = os.path.join(mask_dir, random_mask[0])
                mask_class = random_mask[0].split('_')[0]

                added_mask_img, mask_yolo_label = self.add_mask(source_path, mask_path, mask_class)
                lines.append(mask_yolo_label)  # 在原始 label的基础上把 mask的 label加上
                img_save = os.path.join(save_img_dir, fname+'_insert')
                img_save_path = img_save + '_1'
                i = 1
                while os.path.isfile(img_save_path+'.png'):
                    i += 1
                    img_save_path = img_save + '_' + str(i)
                img_save_path = img_save_path + '.png'
                # cv2.imwrite(img_save_path, added_mask_img)
                cv2.imencode('.png', added_mask_img)[1].tofile(img_save_path)# 增加了mask的图片存入图片文件夹
                label_save_path = os.path.join(save_label_dir, fname + '_' + str(i) + '.txt')
                with open(label_save_path, "w") as f:  # 增加了 mask的新 label存入txt文件夹
                    f.writelines(lines)

    def remove_obj(self):
        img_dir = os.path.join(self.dataset_path, 'images')
        label_dir = os.path.join(self.dataset_path, 'labels')
        save_img_dir, save_label_dir = self.save_dir()
        save_img_tmp_dir = os.path.join(self.ext_path, 'img_tmp')
        os.makedirs(save_img_tmp_dir, exist_ok=True)

        self.get_coco(img_dir, label_dir, save_img_tmp_dir, save_img_dir, save_label_dir)

        command = "python D:/PycharmProjects/objAug_tool/server/lama/bin/predict.py model.path=D:/PycharmProjects/objAug_tool/server/lama/big-lama indir="+save_img_tmp_dir+" outdir="+save_img_dir+" device=cpu"
        print(command)
        os.system(command)
        print("ok")



    def get_coco(self, img_dir, label_dir, save_img_tmp_dir, save_img_dir, save_label_dir):
        for img_file in os.listdir(img_dir):
            fname = img_file.split('.jpg')[0]
            if os.path.exists(os.path.join(label_dir, fname + '.txt')):  # 找到匹配的 jpg 和 txt
                jpg_path = os.path.join(img_dir, img_file)
                txt_path = os.path.join(label_dir, fname + '.txt')
                f = open(txt_path)
                lines = f.readlines()  # 原始 label
                # 保存 mask png图片
                print(save_img_tmp_dir)
                mask_save_path = os.path.join(save_img_tmp_dir, fname + '_mask.png')
                img_save_path = os.path.join(save_img_tmp_dir, fname + '.png')

                if len(lines) > 1:  # label 框多于1个的时候才做删除，不然就变成没有label的图片了
                    self.gen_mask_1(jpg_path, txt_path, img_save_path, mask_save_path, random_mask=True, mask_id=-1)
                    with open(os.path.join(save_label_dir, fname + '_mask.txt'), "w") as f:  # 保存新的 label txt
                        f.writelines(lines[1:])
                else:  # label 框只有1个的时候，直接把原始图片和label保存到对应的文件夹
                    img_save_path = os.path.join(save_img_dir, fname + '.png')
                    img = cv2.imread(str(jpg_path))
                    # cv2.imwrite(img_save_path, img)
                    cv2.imencode('.png', img)[1].tofile(img_save_path)
                    cv2.destroyAllWindows()
                    with open(os.path.join(save_label_dir, fname + '.txt'), "w") as f:  # 保存 label txt
                        f.writelines(lines)

    def gen_mask_1(self, image_path, label_path, img_save_path, mask_save_path, random_mask, mask_id):
        # 读取图像文件
        img = cv2.imread(str(image_path))
        # 读取 label
        with open(label_path, 'r') as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

        if random_mask:
            area_li = []
            for i in range(len(lb)):
                x = lb[i]
                label, x, y, w, h = x
                area_li.append(w * h)
            mask_id = area_li.index(min(area_li))
            x = lb[mask_id]  # 第一个 object label对应的位置信息
            self.mask(img, x, img_save_path, mask_save_path)
        else:
            x = lb[mask_id]  # 第一个 object label对应的位置信息
            self.mask(img, x, img_save_path, mask_save_path)

    def mask(self, img, x, img_save_path, mask_save_path):
        # cv2.imwrite(img_save_path, img)
        cv2.imencode('.png', img)[1].tofile(img_save_path)
        print(img_save_path)
        cv2.destroyAllWindows()

        h1, w1 = img.shape[:2]
        label, x, y, w, h = x

        # 边界框反归一化
        x_t = x * w1
        y_t = y * h1
        w_t = w * w1
        h_t = h * h1

        # 计算坐标
        top_left_x = x_t - w_t / 2
        top_left_y = y_t - h_t / 2
        bottom_right_x = x_t + w_t / 2
        bottom_right_y = y_t + h_t / 2
        # print("左上x坐标:{}".format(top_left_x))
        # print("左上y坐标:{}".format(top_left_y))
        # print("右下x坐标:{}".format(bottom_right_x))
        # print("右下y坐标:{}".format(bottom_right_y))

        black = np.zeros((h1, w1))  # 全黑底色图片，与原图的尺寸一样大
        white_color = (255, 255, 255)
        cv2.rectangle(black, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)),
                      color=white_color, thickness=-1)
        # cv2.imshow('show', black)
        # cv2.imwrite(mask_save_path, black)
        cv2.imencode('.png', black)[1].tofile(mask_save_path)
        # cv2.waitKey(0)  # 按键结束
        cv2.destroyAllWindows()














