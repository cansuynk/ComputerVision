# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:11:54 2019

@author: yanik
"""

#from part1 import sobel_filter
import cv2
import pyautogui
import time
import numpy as np

def canny_edge_detector(image):
   
    row, col, ch = image.shape
    
    #Since Canny is sensitive to noise, first apply smoothing by using Gaussian kernel
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image,(5,5),0)
    
    #I used threshold method to find optimum threshold and lower one is 3 times less than max threshold
    max_treshold,img = cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_treshold = max_treshold/3
    
    #Apply Canny filter
    image = cv2.Canny(image,low_treshold,max_treshold)
    
    #I applied contour and draw it
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    image = cv2.drawContours(image, contours, -1, (255,255,255), 1)
        
    return image

'''
#This part is testing Part2
time.sleep(5)

#Takes screenshot, saves it and loads
myScreenshot = pyautogui.screenshot()
myScreenshot.save('test.png')
image = cv2.imread("./test.png")

#This part crops desired area from the whole screenshot
row, col, ch = image.shape
upLeft = (int(col / 4), int(row / 4.5))
bottomRight = (int(col * 3 / 4), int(row * 3 / 4))

image = image[upLeft[1]:bottomRight[1], upLeft[0]:bottomRight[0]]



image = canny_edge_detector(image) 
#image =  cv2.add(image, image2)
cv2.imwrite("Part2.jpg", image)
cv2.imshow('Output', image)
cv2.waitKey()
'''