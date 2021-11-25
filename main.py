import cv2
import numpy as np
# read images

img = cv2.imread('d.jpg', cv2.IMREAD_UNCHANGED)
#img= cv2.medianBlur(img, 5)
img = cv2.blur(img,(5,5))

# Binary
ret, thresh = cv2.threshold(
    cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY),  # Change into Grayscale image
    150, 255,   # Change the pixels bigger than 150 into 255  otherwise put it into 0
    cv2.THRESH_BINARY)  # Binary as black and white
#cv2.imshow("contours", thresh)

# Search boundry
contours, hierarchy = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)


for c in contours:

    x, y, w, h = cv2.boundingRect(c)

    """
    Put into a boundry, x and y are the points of up-left, w and h are width and length of the rectangle
    """
    if w>200:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    """
    draw the rectangle
        img is the original image to draw the outline
        (x, y) is the coordinate of the point in the upper left corner
        (x-w, y-h) is the coordinate in the lower right corner
        (0,255,0）it the color of line
        2 is the width of line
    """

    # Get the smallest rectangular profile May have a rotation angle
    rect = cv2.minAreaRect(c)
    # Calculates the coordinates of the smallest area
    box = cv2.boxPoints(rect)
    # nomalize the coordinates into int
    box = np.int0(box)
    print(box)
    bb = box[0,0]
    aa = box[2,0]


    # draw the outlines
    if (aa - bb)>300:
        cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

    # Calculates the center and radius of the smallest closed circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    # Change into int
    center = (int(x), int(y))
    radius = int(radius)
    #print(radius)
    # draw circles
    if radius>100:
        img = cv2.circle(img, center, radius, (0, 255, 0), 2)

# draw outlines
cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
cv2.imshow("contours", img)
cv2.waitKey()
cv2.destroyAllWindows()