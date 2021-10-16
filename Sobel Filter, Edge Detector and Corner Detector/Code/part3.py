# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:29:25 2019

@author: yanik
"""
import cv2
import pyautogui
import time
import numpy as np
from part1 import sobel_filter

def min_eigenvalue_corner_detector(image):

    row, col, ch = image.shape
    
    #Apply sobel filter to take derivatives
    sobelx, sobely = sobel_filter(image)
    
    #To be able to get image structure tensor(2x2), convolve each values with Gaussian kernel
    M_00 = cv2.GaussianBlur(sobelx*sobelx,(5,5),0)
    M_01 = cv2.GaussianBlur(sobelx*sobely,(5,5),0)
    M_11 = cv2.GaussianBlur(sobely*sobely,(5,5),0)
    
    #minimum eigen value calculation (I got a reference from our lecture slides)
    min_eigen_values = 0.5 * ((M_00 + M_11) - ((M_00 - M_11)**2 + 4*(M_01**2))**0.5)
    #min_eigen_values = cv2.dilate(min_eigen_values,None)
    

    
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    
    #I applied a threshold to take sharp corners and changed the colors
    image[min_eigen_values>0.15*min_eigen_values.max()]=[0,0,255]
     
    #Holds corners coordinates
    corner_x = []
    corner_y = []
      
    #To be able to see corners clearly, I traversed pixels, found red pixels, took coordinates and drew rectangles
    for i in range(row):
        for j in range(col):
            if (image[i][j][0] == 0 and image[i][j][1] == 0 and image[i][j][2] == 255):
                corner_x.append(j)
                corner_y.append(i)
                image = cv2.rectangle(image, (j, i), (j + 5, i + 5), (0, 255, 0), -1)
    
    '''
    #You can open comments to show and save output
    cv2.imshow('Output', image)
    #cv2.imwrite("Part3.jpg", image)
    cv2.waitKey()
    '''
    return corner_x, corner_y

'''
#This part is for testing
time.sleep(5)

myScreenshot = pyautogui.screenshot()
myScreenshot.save('test.png')

image = cv2.imread("./test.png")
row, col, ch = image.shape

#This part crops desired area from the whole screenshot
upLeft = (int(col / 4), int(row / 4.5))
bottomRight = (int(col * 3 / 4), int(row * 3 / 4))

image = image[upLeft[1]:bottomRight[1], upLeft[0]:bottomRight[0]]

corner_x, corner_y = min_eigenvalue_corner_detector(image) 
'''