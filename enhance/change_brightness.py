# encoding:gbk
from skimage import exposure,io
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
            name3 = (str(name) + '_dark_' + str(i) + '.jpg')
            flag+=1
            print(flag)
            print(name3)
            img=io.imread(ORIGIN_DIR +"/" + name1)

            gam1 = exposure.adjust_gamma(img, 2)

            copy_to_dir = os.path.join(COPY_DIR, names)
            io.imsave(copy_to_dir +"/"+ name3,gam1)
            print('saved')

            name2 = (str(name) + '_light_' + str(i) + '.jpg')
            flag+=1
            print(flag)
            print(name2)
            img=io.imread(ORIGIN_DIR_45 + "/" + name1)
            gam2 = exposure.adjust_gamma(img, 0.5)

            copy_to_dir = os.path.join(COPY_DIR, names)
            io.imsave(copy_to_dir +"/" + name2,gam2)
            print('saved')




