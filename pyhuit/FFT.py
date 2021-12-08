import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


input_path = 'C:/Users/Tardis/Desktop/huit/data_green'
for root, dirs, files in os.walk(input_path):
    for name in files:
        file = os.path.join(root, name)



        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        img = 255 - img
        ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        # #cv2.imshow("contours", thresh)

        # Search boundry
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # draw outlines
        #cv2.namedWindow("contours", 0);
        #cv2.resizeWindow("contours", 1920, 1080);
        cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
        #cv2.imshow("contours", img)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

        #img = cv2.imread("C:/Users/Tardis/Desktop/huit/data_green/green001c.png", 0)
        img = cv2.imread(file, 0)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)



        rows,cols = img.shape
        crow,ccol = int(rows/2), int(cols/2)
        fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
        print(fshift)
        ishift = np.fft.ifftshift(fshift)
        iimg = np.fft.ifft2(ishift)
        iimg = np.abs(iimg)
        height = iimg.shape[0]
        weight = iimg.shape[1]


        #print(height)
        #print(weight)
        q = 0
        p = 0
        m = 0
        for row in range(height):  # 遍历高
            for col in range(weight):  # 遍历宽
                dis = cv2.pointPolygonTest(contours[0], (col, row), False)
                if int(dis) >= 0:
                    p = p+iimg[row,col]
                    m = m + (iimg[row,col])*(iimg[row,col])
                    q=q+1
        #print(q)
        #print(p)
        #print(m)
        print(name+","+str(int(m/q - (p/q)*(p/q))))






        plt.subplot(211)

        plt.imshow(img.astype('uint8'), cmap = 'gray')

        plt.title('original')
        plt.axis('off')

        plt.subplot(212)

        plt.imshow(iimg.astype('uint8'), cmap = 'gray')

        plt.title('FFT')
        plt.axis('off')


        plt.show()
