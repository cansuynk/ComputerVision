# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 00:03:19 2019

@author: yanik
"""



### merkezi pathe göre almaya çalış
### sağ olayını düzelt

### imagedeki köşeleri ekle
import cv2
import pyautogui
import time
import numpy as np
from part2 import canny_edge_detector
from part3 import min_eigenvalue_corner_detector



def largets_point(image):

        
        #Finds the center of image, this coordinate will be used for distance calculation
        row, col, ch = image.shape
        centerx = int(col/2)
        centery = int(row/2)
        
        #Finds image eigenvalues of image
        corner_x, corner_y = min_eigenvalue_corner_detector(image)
        
        for i in range(len(corner_x)):
            image = cv2.rectangle(image, (corner_x[i], corner_y[i]), (corner_x[i] + 5, corner_y[i] + 5), (0, 255, 0), -1)
            
        
        #Draw center of image
        image = cv2.rectangle(image, (centerx, centery), (centerx + 5, centery + 5), (0, 0, 255), -1)
        
        
        #Using corner points, finds the farthest point from the center coordinate
        #Since my algorithm finds also 4 corner of the image shape (0-0, 0-row, col-0, col-row), I checked them to ignore
        large_distance = 0
        
        for i in range(len(corner_x)):
            if(corner_x[i] != 0 and corner_y[i] !=0) and (corner_x[i] != 0 and corner_y[i] !=row) and (corner_x[i] != col and corner_y[i] !=0) and (corner_x[i] != col and corner_y[i] !=row):
                distance = ((corner_x[i] - centerx)**2 + (corner_y[i] - centery)**2)**0.5
                if distance > large_distance:
                    if centery - corner_y[i] >= 0:
                        large_distance_x = corner_x[i]
                        large_distance_y = corner_y[i]
                        large_distance = distance
        
        #Draws the found point
        image = cv2.rectangle(image, (large_distance_x, large_distance_y), (large_distance_x + 8, large_distance_y + 8), (255, 255, 255), -1)   
        
        '''
        #You can open comments to see output
        cv2.imshow('Output', image)
        cv2.waitKey()
        '''
        
        return large_distance_x, large_distance_y



#Game can finish in two step 
#I recommend to wait between two step. Because it sleeps to take screenshot
for i in range(2):
    
    time.sleep(5)
    #Takes the screenshot, saves and reads
    myScreenshot = pyautogui.screenshot()
    myScreenshot.save('test.png')
    
    image = cv2.imread("./test.png")
   
    #Finds center coordinates of image
    row, col, ch = image.shape
    centerx = int(col/2)
    centery = int(row/2)
    
    #This part crops desired area from the whole screenshot
    upLeft = (int(col / 4), int(row / 4.5))
    bottomRight = (int(col * 3 / 4), int(row * 3 / 4))
    image = image[upLeft[1]:bottomRight[1], upLeft[0]:bottomRight[0]]
    
    #Finds larger distance between the corner and center coordinates
    large_distance_x, large_distance_y = largets_point(image)
    
    #run the monster to a proporsion of the distance
    for i in range(int((centery - large_distance_y)*0.04)):
        pyautogui.keyDown('shift')
        pyautogui.keyDown('w')
           
    pyautogui.keyUp('shift')
    pyautogui.keyUp('w')
    
    #This if condition is for making stop the monster when it reaches the end.
    if (((centerx - large_distance_x)*0.04) < 11):
        for i in range(int((centerx - large_distance_x)*0.06)):
            pyautogui.keyDown('shift')
            pyautogui.keyDown('d')
    
    pyautogui.keyUp('shift')
    pyautogui.keyUp('d')
        


