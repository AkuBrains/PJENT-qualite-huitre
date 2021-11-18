# encoding:gbk
from skimage import io
import skimage
import os
import random
import math


ROOT_DIR = os.path.abspath("C:/Users/theDoctor/Desktop/orville/������������/UNetLSTM-master/images/paris/")
COPY_DIR = os.path.abspath("C:/Users/theDoctor/Desktop/orville/������������/UNetLSTM-master/images/paris/")
ORIGIN_DIR = os.path.abspath("C:/Users/theDoctor/Desktop/orville/������������/UNetLSTM-master/images/paris/")
ORIGIN_DIR_45 = os.path.abspath("C:/Users/theDoctor/Desktop/orville/������������/UNetLSTM-master/images/paris/")


classes = ["ɽ","չ��","��","ͥԺ","��","�㳡","������","��·","̨ͤ¥��","����",
           "�����","���","ǰ̨","����","����","��԰","��԰","����","կ","��Ϣ��","��̨"]

for names in classes:
    flag = 0
    print(names)
    img_path = os.path.join(ROOT_DIR, names)
    imglist = os.listdir(img_path)
    print(len(imglist))
    random_imglist = random.sample(imglist, math.ceil((100 - len(imglist))/4))

    for file_name in random_imglist:
        for i in range(4):
            name = os.path.splitext(file_name)[0]
            name1 = (str(name) + '_' + str(i) + '.jpg')
            name3 = (str(name) + '_pepper_' + str(i) + '.jpg')
            flag+=1
            print(flag)
            print(name3)
            img=io.imread(ORIGIN_DIR +"/" + name1)

            img1 = img / 255.00
            img2 = skimage.util.random_noise(img1, mode='pepper', seed=None, clip=True)

            copy_to_dir = os.path.join(COPY_DIR, names)
            io.imsave(copy_to_dir +"/"+ name3,img2)
            print('saved')

            name2 = (str(name) + '_salt_' + str(i) + '.jpg')
            flag+=1
            print(flag)
            print(name2)
            img=io.imread(ORIGIN_DIR_45 + "/" + name1)
            img1 = img / 255.00
            img2 = skimage.util.random_noise(img1, mode='salt', seed=None, clip=True)

            copy_to_dir = os.path.join(COPY_DIR, names)
            io.imsave(copy_to_dir +"/" + name2,img2)
            print('saved')


