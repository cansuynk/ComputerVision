# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:59:16 2019

@author: yanik
"""

import cv2
import pyautogui
import time
import numpy as np
import matplotlib.pyplot as plt


#You can take screenshot
#Takes screenshot, saves it and loads
time.sleep(5)

myScreenshot = pyautogui.screenshot()
myScreenshot.save('test.png')


image = cv2.imread("./test.png")

#This part crops desired area from the whole screenshot
row, col, ch = image.shape
centerx = int(col/2)
centery = int(row/2)

upLeft = (int(col / 4), int(row / 4.5))
bottomRight = (int(col * 3 / 4), int(row * 3 / 4))
image = image[upLeft[1]:bottomRight[1], upLeft[0]:bottomRight[0]]

#Convert image to grayscale 
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

histogram = cv2.calcHist([image],[0],None,[256],[0,256])

'''
#You can plot the histogram
plt.plot(histogram)
plt.xlim([0,256])
'''

#Function takes a range of the treshold and calculates mean weight and variance
def calc_variance(start, end):
    
    #Finds total intensity
    total_pixel = 0
    for i in range(len(histogram)):
        total_pixel = total_pixel + histogram[i]
    
    #canculates weight and mean
    W = 0
    M = 0
    Mean = 0
    for i in range(start, end):
        W = W + histogram[i]
        M = M + i*histogram[i]
    Weight = W / total_pixel
    if W!=0: Mean = M / W
    
    #Calculates variance
    V = 0
    Variance = 0
    for i in range(start, end):
        V = V + (((i - Mean)**2) * histogram[i])
    if W!=0: Variance = V / W    
    return Weight, Mean, Variance
    
#For each threshold value calc_variance function is called and I try to find minimum within variance
threshold = 0
max_between_class =  0
between_class = 0;
for i in range(0,256):
    
    #Calculate weight, mean and variance for both background and foreground
    Wb, Mb, Vb = calc_variance (0,i+1)
    Wf, Mf, Vf = calc_variance (i+1,256)   
    
    #Calculates between variance, and try to find maximum one to find optimum threshold
    between_class = Wb * Wf * (Mb - Mf)**2
    if (between_class > max_between_class):
        max_between_class = between_class
        threshold = i


row, col = image.shape
output1 = np.zeros((row,col,1), np.uint8)

#Apply the threshold to image
for y in range(row):
    for x in range(col):
        # threshold the pixel
        pixel = image[y, x]
        output1[y, x] = 0 if pixel < threshold else pixel
        
#Also by using existing threshold methold find optimum one
threshold2, output2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
#Print boths and compare them
print(threshold, threshold2)
print(max_between_class)

'''
#Also segmented images can be compared
cv2.imshow('output1', output1)
cv2.imwrite("Part5_1.jpg", output1)
cv2.imshow('output2', output2)
cv2.imwrite("Part5_2.jpg", output2)
cv2.waitKey()
'''    
    











