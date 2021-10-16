# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math


def catLandmarks():
    templatePoints = np.load("template_points.npy")     #open files

    catImage = cv2.imread("./CAT_00/00000095_001.jpg")
    rows, cols, ch = catImage.shape
    
    
    file = open("./CAT_00/00000095_001.jpg.cat")       #Open points file and takes the points
    content = file.read()
    file.close()
    
    content = content.split()                   #parse according to space
    content = [int(i) for i in content]         #convert string values to integer
    
    index = 1
    for i in range (0, content[0]):             #draws points on the cat image
        x = content[index]
        y = content[index+1]
        index = index + 2
        catImage = cv2.rectangle(catImage, (x, y), (x + 5, y + 5), (0, 255, 0), -1)
    
    ##########################################################################################################
      
    
    leftEye_x = int((templatePoints[36][0] + templatePoints[39][0])/2)      #Finds left eye x coordinate
    leftEye_y = int((templatePoints[36][1] + templatePoints[39][1])/2)      #Finds left eye y coordinate
    
    rightEye_x = int((templatePoints[42][0] + templatePoints[45][0])/2)     #Finds right eye x coordinate
    rightEye_y = int((templatePoints[42][1] + templatePoints[45][1])/2)     #Finds right eye y coordinate
    
    distanceEyes_template = math.sqrt((rightEye_x - leftEye_x)**2 + (rightEye_y - leftEye_y)**2)    #Distance between template eyes
    distanceEyes_cat = math.sqrt((content[1] - content[3])**2 + (content[2] - content[4])**2)       #Distance between cat eyes
    
    horizontalRatio = distanceEyes_template/distanceEyes_cat    
    
    ############################################################################################################
    midEyes_template_x = int((leftEye_x + rightEye_x)/2)        #Finds middle x coordinate between template eyes
    midEyes_template_y = int((leftEye_y + rightEye_y)/2)        #Finds middle y coordinate between template eyes
    
    #Distance between midpoint of eyes and the mouth
    distanceMouth_template = math.sqrt((midEyes_template_x - templatePoints[66][0])**2 + (midEyes_template_y - templatePoints[66][1])**2)
    
    
    midEyes_cat_x = int((content[1] + content[3])/2)        #Finds middle x coordinate between cat eyes
    midEyes_cat_y = int((content[2] + content[4])/2)        #Finds middle y coordinate between cat eyes
    
    #Distance between midpoint of eyes and the mouth
    distanceMouth_cat = math.sqrt((midEyes_cat_x - content[5])**2 + (midEyes_cat_y - content[6])**2)
    
    verticalRatio = distanceMouth_template/distanceMouth_cat
    
    #############################################################################################################
    
    for i in range(0, 68):          #Apply ratios to each point
        x = int(templatePoints[i][0] * horizontalRatio)
        y = int(templatePoints[i][1] * verticalRatio)

        templatePoints[i][0] = x
        templatePoints[i][1] = y
        
    #Finds new left and right eyes to use for transformation matrix (To be able to locate point I apply transformation two times)
    for a in range(2):
        
        #Finds new left and right eyes
        left_x = int((templatePoints[36][0] + templatePoints[39][0])/2)     
        left_y = int((templatePoints[36][1] + templatePoints[39][1])/2)
        
        right_x = int((templatePoints[42][0] + templatePoints[45][0])/2)
        right_y = int((templatePoints[42][1] + templatePoints[45][1])/2)
        
        #Reference points
        pts1 = np.float32([[content[1],content[2]],[content[3],content[4]],[content[5],content[6]]])
        pts2 = np.float32([[left_x,left_y],[right_x,right_y],[templatePoints[66][0],templatePoints[66][1]]])  
        
        #Transformation matrix
        M = cv2.getAffineTransform(pts2,pts1)
        
        #Apply transformation matris to templete points
        templatePoints = np.matmul(templatePoints[:,[0,1]], M)
    
    catLandmark_x = []
    catLandmark_y = []
    
    #To be able to locate points on cat face, I need to shift points and save them
    for i in range(0, 68):
        x = int(templatePoints[i][0] + 45)
        y = int(templatePoints[i][1] + 75)
        
        catLandmark_x.append(x)
        catLandmark_y.append(y)
        
        catImage = cv2.rectangle(catImage, (x, y), (x + 5, y + 5), (255, 0, 0), -1)
    
    #To display the image you can open the comments
    """
    cv2.imshow("Cat", catImage)
    cv2.waitKey(0)
    """
    
    #To save the image you can open the comments
    """
    cv2.imwrite("Part2_cat.jpg", catImage)
    """
    
    return catLandmark_x, catLandmark_y
    

catLandmarks()

