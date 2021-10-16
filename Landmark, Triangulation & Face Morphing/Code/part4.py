# -*- coding: utf-8 -*-


import cv2
import numpy as np
import moviepy.editor as mpy

image = np.zeros((500,375,3), np.uint8)        #Create image to display source triangle
image2 = np.zeros((500,375,3), np.uint8)       #Create image to display target triangle

rows, cols, ch = image.shape

#Source triangle vertices
source_points = [[100, 200], [200, 50], [50, 100]]
source_points = np.int32([source_points])

#Target triangle vertices
target_points = [[300, 350], [350, 250], [150, 200]]
target_points = np.int32([target_points])

#Source image
cv2.polylines(image, [source_points], isClosed=True, color=(0,0,255), thickness=1)
cv2.fillPoly(image, [source_points], color=(0,0,255))

#Target image
cv2.polylines(image2, [target_points], isClosed=True, color=(255,0,0), thickness=1)
cv2.fillPoly(image2, [target_points], color=(255,0,0))


images_list = []    #Holds video frames

stepNumber = 20

for i in range(0,stepNumber):

    #In each step, I find a ratio(alpha) which is used to find morhed image and to change the intensity between source and target images 
    alpha = i/(stepNumber-1)
    
    xm = []
    ym = []
    
    #Each step, calculate the location (xm, ym) of the pixel in the morphed image
    for i in range(0, 3):
        x = (1-alpha)*source_points[0][i][0] + alpha*target_points[0][i][0]
        y = (1-alpha)*source_points[0][i][1] + alpha*target_points[0][i][1]
        
        xm.append(x)
        ym.append(y)
    
    #Points of source and target image
    pts1 = np.float32([[source_points[0][0][0],source_points[0][0][1]],[source_points[0][1][0],source_points[0][1][1]],[source_points[0][2][0],source_points[0][2][1]]])
    pts2 = np.float32([[target_points[0][0][0],target_points[0][0][1]],[target_points[0][1][0],target_points[0][1][1]],[target_points[0][2][0],target_points[0][2][1]]])
    

    #Converts morphed image array to proper type to apply affine transform
    morphed = [[xm[0], ym[0]], [xm[1], ym[1]], [xm[2], ym[2]]]
    morphed =  np.float32(morphed)
    
    #Apply affine transform to both source and target images
    transformationMatrix = cv2.getAffineTransform(pts1,morphed)
    warp1 = np.matmul(pts1, transformationMatrix) 
    
    transformationMatrix = cv2.getAffineTransform(pts2,morphed) 
    warp2 = np.matmul(pts2, transformationMatrix)
    
    #Changes the intensity between source and target warp images according to alpha
    warpImage = (1.0 - alpha) * warp1 + alpha * warp2
    
    image3 = np.zeros((500,375,3), np.uint8)        #Creates new image to save new triangle
    
    points = [[warpImage[0][0], warpImage[0][1]], [warpImage[1][0], warpImage[1][1]], [warpImage[2][0], warpImage[2][1]]]
    points = np.int32([points])
    
    #Draw new triangle, also use alpha for changing the color
    cv2.polylines(image3, [points], isClosed=True, color=(0+alpha*255,0,255-alpha*255), thickness=1)
    cv2.fillPoly(image3, [points], color=(0+alpha*255,0,255-alpha*255))
    

    images_list.append(image3)      #Saves the image to use for video
    
#Since moviepy works with RGB images, I converted them into RGB
for i in range(len(images_list)):
    images_list[i] = cv2.cvtColor(images_list[i], cv2.COLOR_BGR2RGB)

#Create gif
clip = mpy.ImageSequenceClip(sequence=images_list, fps=25) 
clip.write_gif('part4_gif.gif')

#To display the source and target triangles, open comments
"""
cv2.imshow("Source Triangle", image)
cv2.imshow("Target Triangle", image3)
cv2.waitKey(0)  
"""