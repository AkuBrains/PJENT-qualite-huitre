# encoding:gbk
#import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import os
import random
import math

ROOT_DIR = os.path.abspath("/code/data/365/origin_25classes/train/")
COPY_DIR = os.path.abspath("/code/data/365/final/train/")
ORIGIN_DIR = os.path.abspath("/code/data/pano80k_cube0/pano80k_cube0/")
ORIGIN_DIR_45 = os.path.abspath("/code/data/pano80k_cube45/pano80k_cube45/")

classes = ["卫生间","山","展厅","湖","庭院","桥","广场","会议室","道路","亭台楼阁","王宫",
           "宴会厅","佛殿","前台","寺庙","包厢","公园","花园","走廊","寨","休息室","阳台"]

for names in classes:
    flag = 0
    print(names)
    img_path = os.path.join(ROOT_DIR, names)
    imglist = os.listdir(img_path)
    print(len(imglist))
    random_imglist = random.sample(imglist, math.ceil((1000 - len(imglist))/4))

    for file_name in random_imglist:
        for i in range(4):
            name = os.path.splitext(file_name)[0]
            name1 = (str(name) + '_' + str(i) + '.jpg')
            name3 = (str(name) + '_hue_' + str(i) + '.jpg')
            flag+=1
            print(flag)
            print(name3)
            img=io.imread(ORIGIN_DIR +"/" + name1)

            img = img * 1.0
            img_out = img * 1.0

            # -1 ~ 1
            Increment = 0.5

            img_min = img.min(axis=2)
            img_max = img.max(axis=2)

            Delta = (img_max - img_min) / 255.0
            value = (img_max + img_min) / 255.0
            L = value/2.0

            mask_1 = L < 0.5

            s1 = Delta/(value + 0.001)
            s2 = Delta/(2 - value + 0.001)
            s = s1 * mask_1 + s2 * (1 - mask_1)

            if Increment >= 0 :
                temp = Increment + s
                mask_2 = temp >  1
                alpha_1 = s
                alpha_2 = s * 0 + 1 - Increment
                alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
                alpha = 1/(alpha + 0.001) -1
                img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
                img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
                img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha

            else:
                alpha = Increment
                img_out[:, :, 0] = L * 255.0 + (img[:, :, 0] - L * 255.0) * (1 + alpha)
                img_out[:, :, 1] = L * 255.0 + (img[:, :, 1] - L * 255.0) * (1 + alpha)
                img_out[:, :, 2] = L * 255.0 + (img[:, :, 2] - L * 255.0) * (1 + alpha)


            img_out = img_out/255.0

            # 饱和处理
            mask_1 = img_out  < 0
            mask_2 = img_out  > 1

            img_out = img_out * (1-mask_1)
            img_out = img_out * (1-mask_2) + mask_2

            img=np.array(img,dtype=np.uint8)
            copy_to_dir = os.path.join(COPY_DIR, names)
            io.imsave(copy_to_dir +"/"+ name3,img_out)
            print('saved')

            name2 = (str(name) + '_45hue_' + str(i) + '.jpg')
            flag+=1
            print(flag)
            print(name2)
            img=io.imread(ORIGIN_DIR_45 + "/" + name1)

            img = img * 1.0
            img_out = img * 1.0

            # -1 ~ 1
            Increment = 0.5

            img_min = img.min(axis=2)
            img_max = img.max(axis=2)

            Delta = (img_max - img_min) / 255.0
            value = (img_max + img_min) / 255.0
            L = value/2.0

            mask_1 = L < 0.5

            s1 = Delta/(value + 0.001)
            s2 = Delta/(2 - value + 0.001)
            s = s1 * mask_1 + s2 * (1 - mask_1)

            if Increment >= 0 :
                temp = Increment + s
                mask_2 = temp >  1
                alpha_1 = s
                alpha_2 = s * 0 + 1 - Increment
                alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
                alpha = 1/(alpha + 0.001) -1
                img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
                img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
                img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha

            else:
                alpha = Increment
                img_out[:, :, 0] = L * 255.0 + (img[:, :, 0] - L * 255.0) * (1 + alpha)
                img_out[:, :, 1] = L * 255.0 + (img[:, :, 1] - L * 255.0) * (1 + alpha)
                img_out[:, :, 2] = L * 255.0 + (img[:, :, 2] - L * 255.0) * (1 + alpha)


            img_out = img_out/255.0

            # 饱和处理
            mask_1 = img_out  < 0
            mask_2 = img_out  > 1

            img_out = img_out * (1-mask_1)
            img_out = img_out * (1-mask_2) + mask_2

            img=np.array(img,dtype=np.uint8)
            copy_to_dir = os.path.join(COPY_DIR, names)
            io.imsave(copy_to_dir +"/" + name2,img_out)
            print('saved')

            #plt.figure()
            #plt.imshow(img/255)
            #plt.axis('off')

            #plt.figure(2)
            #plt.imshow(img_out)
            #plt.axis('off')

            #plt.show()
