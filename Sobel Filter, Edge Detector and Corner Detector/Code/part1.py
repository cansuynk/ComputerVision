# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:20:02 2019

@author: yanik
"""


import pyautogui
import time

import cv2
import numpy as np
#from part2 import canny_edge_detector

from scipy.signal import convolve2d

def sobel_filter(image):
    
    row, col, ch = image.shape
 
    #Convert grayscale
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)   

    #Gaussian Smoothing
    image = cv2.GaussianBlur(image,(5,5),0)
    
    
    #vertical and horizontal masks
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    #Holds convolution results, matrices
    sobelx = np.zeros((row,col,1), np.uint8)
    sobely = np.zeros((row,col,1), np.uint8)
    
    #Padding for convolution
    image_temp = np.zeros((row + 2, col + 2))  
    image_temp[1:-1, 1:-1] = image
    
    #Convolve image with kernels
    sobelx = convolve2d(image_temp, Gx, mode='same', boundary='fill', fillvalue=0)   
    sobely = convolve2d(image_temp, Gy, mode='same', boundary='fill', fillvalue=0)  
   
    
    '''
    #I also added my own convolution steps but it makes operations slow
    #Thats why I used existing convolve function.
    
    m, n = Gx.shape
    y, x = image.shape
    y = y - m + 1
    x = x - m + 1
    #sobelx = np.zeros((y,x,1), np.uint8)
    #sobely = np.zeros((y,x,1), np.uint8)
    for i in range(y):
        for j in range(x):
            sobelx[i][j] = np.sum(image[i:i+m, j:j+m]*Gx)
            sobely[i][j] = np.sum(image[i:i+m, j:j+m]*Gy)
   
    '''
    #Remove padding part
    sobelx = sobelx[1:-1, 1:-1]
    sobely = sobely[1:-1, 1:-1]
    
    #Edge magnitude
    image = (sobelx*sobelx + sobely*sobely)**0.5
    
    '''
    #You can open comments to see and save the image
    cv2.imshow('Output', image)
    cv2.waitKey()
    cv2.imwrite("Part1.jpg", image)
    '''
    
    return sobelx, sobely

'''
#This part is for testing Part1

time.sleep(5)
#Takes screenshot, saves it and loads
myScreenshot = pyautogui.screenshot()
myScreenshot.save('test.png')

image = cv2.imread("./test.png")
row, col, ch = image.shape

#This part crops desired area from the whole screenshot
upLeft = (int(col / 4), int(row / 4.5))
bottomRight = (int(col * 3 / 4), int(row * 3 / 4))

image = image[upLeft[1]:bottomRight[1], upLeft[0]:bottomRight[0]]

#image2 = canny_edge_detector(image) 
#image =  cv2.add(image, image2)

sobelx, sobely = sobel_filter(image)
'''
