# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 01:06:14 2019

@author: yanik
"""

import nibabel as nib
import numpy as np


threshold = 10

def regionGrowing(img, seed, neighborhood):
    
    row, col = img.shape
    size = row * col
    
    #If 4 neighborhood, uses corresponding points
    if (neighborhood == 4):
        neighborPts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        steps = 4
    elif (neighborhood == 8):
        neighborPts = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
        steps = 8
    
    #result image
    segmentedImg = np.zeros((row, col, 1), np.uint8)
    regionSize = 1
    
    #neighbors points
    neighbors_x = []
    neighbors_y = []
    neighborIntensities = []

    #seed intensity
    mean = img[seed]
    #intensity difference
    difference=0

    while (regionSize < size):
        
        if(difference > threshold):
            break
        
        for i in range(steps):
            #Finds neighbor points
            x = neighborPts[i][0] + seed[0]
            y = neighborPts[i][1] + seed[1]

            #If point is in the image (checks image borders)
            if ((x >= 0) and (x < row) and (y >= 0) and (y < col)):
                    #Prevent looking previous point
                    if segmentedImg[x, y] == 0:
                        neighbors_x.append(x)
                        neighbors_y.append(y)
                        neighborIntensities.append(img[x, y])
                        segmentedImg[x, y] = 255
                        
        #Find min distance between intensities
        minDistance = abs(neighborIntensities[0]-mean)
        i=1
        index=0
        for i in range(len(neighborIntensities)):
            distance = abs(neighborIntensities[i] - mean)
            #Finds that point index
            if distance < minDistance:
                minDistance = distance
                index = i
         
        difference = minDistance
        
        #Finds new mean
        total = mean*regionSize + neighborIntensities[index]
        mean = total/regionSize
        
        #Finds next seed point
        seed = [neighbors_x[index], neighbors_y[index]]
        #Remove that point from array
        neighborIntensities[index] = neighborIntensities[-1]
        neighbors_x[index] = neighbors_x[-1]
        neighbors_y[index] = neighbors_y[-1]
        
        regionSize = regionSize+1

    return segmentedImg