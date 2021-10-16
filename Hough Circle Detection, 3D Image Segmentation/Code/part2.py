# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 16:36:03 2019

@author: yanik
"""

import cv2
from math import sqrt, pi, cos, sin
import numpy as np

image = cv2.imread("./Material/marker.png")
row, col, ch = image.shape


#Apply Gaussian Blur
blur = cv2.GaussianBlur(image,(5,5),0)
#Convert image to grayscale
blurgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

'''
#This part is written by using cv2's Hough method
d_length = int(((row**2 + col **2)**(1/2))/4)

#Apply Hough method
circles = cv2.HoughCircles(blurgray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, 
                           minRadius=1, maxRadius=d_length)

#Displays circles
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    center = (i[0], i[1])
    # circle center
    cv2.circle(image, center, 2, (0, 0, 255), 3)
    # circle outline
    radius = i[2]
    cv2.circle(image, center, radius, (0, 255, 0), 3)
    
    
cv2.imshow("detected circles", image)
cv2.waitKey(0)

'''
#Find thresholds
max_treshold,image1 = cv2.threshold(blurgray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
low_treshold = max_treshold/3
#Apply canny operator
image2 = cv2.Canny(image1,low_treshold,max_treshold)

'''
cv2.imshow("canny", image2)
cv2.waitKey(0)
'''

#Create accumulator
A = np.zeros((row+10,col+10,30), dtype = np.float32)

for x in range(row):
    for y in range(col):
        if(image2[x,y] > 0):
            for r in range(20, 30):
                for t in range(360):
                     a = int(x-r * cos(t * (pi/180)))
                     b = int(y-r * sin(t * (pi/180)))
                     A[a][b][r] += 1


image3 = cv2.imread("./Material/marker.png")
for a in range(col):
    for b in range(row):
        for r in range(20, 30):
            if (A[b][a][r] > 150):
                cv2.circle(image3, (a,b), 2, (0, 0, 255), 3)
                cv2.circle(image3, (a,b), r, (0, 255, 0), 3)


cv2.imshow("out", image3)
cv2.imwrite("part2.jpg", image3)
cv2.waitKey(0)

