# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 22:12:14 2019

@author: yanik
"""


import cv2
import nibabel as nib
from regionGrowing import regionGrowing 
import numpy as np


img = nib.load('./Material/V_seg_05.nii')


row, col, z_index = img.shape


found = False

#For each slice find a white pixel
for z in range(z_index):
    for i in range(row):
        for j in range(col):
            if img.get_fdata()[i,j,z] > 0:
                if found==False:
                    found = True
                    seed = (i,j)
                    
    #If white pixel is found, calls regiongrowing function, and saves images
    if(found == True):
        output = regionGrowing(img.get_fdata()[:,:,z], seed, 8)
        found = False
        print(str(z) + " Processing...")
        #cv2.imwrite("./N-8_Segmentations/8N_"+str(z)+".jpg", output) 

    if (z == 46):
        seg = output


gt = img.get_fdata()[:,:,46]
dice = np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt))
print ('Dice similarity score is {}'.format(dice))


##### 4-neighborhood #####
found = False

for z in range(z_index):
    for i in range(row):
        for j in range(col):
            if img.get_fdata()[i,j,z] > 0:
                if found==False:
                    found = True
                    seed = (i,j)
                    
    
    if(found == True):
        output = regionGrowing(img.get_fdata()[:,:,z], seed, 4)
        found = False
        print(str(z) + " Processing...")
        #cv2.imwrite("./N-4_Segmentations/4N_"+str(z)+".jpg", output)      
    
    if (z == 46):
        seg2 = output



gt2 = img.get_fdata()[:,:,46]
dice2 = np.sum(seg2[gt2==1])*2.0 / (np.sum(seg2) + np.sum(gt2))
print ('Dice similarity score is {}'.format(dice2))


