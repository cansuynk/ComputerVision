# -*- coding: utf-8 -*-

import cv2
import numpy as np
import moviepy.editor as mpy

from part2 import catLandmarks
from part3 import subdivPoints, landmarks

#Loads images
catImage = cv2.imread("./CAT_00/00000095_001.jpg")
rows, cols, ch = catImage.shape

firstImage = cv2.imread("./dennis_ritchie.jpg")
secondImage = cv2.imread("./yusuf.jpg")

#Since it is necessary for all three photos to be same size, I resized the photos
firstImage = cv2.resize(firstImage,(cols,rows))
secondImage = cv2.resize(secondImage,(cols,rows))


def morphing(landmarks_x_1, landmarks_y_1, landmarks_x_2, landmarks_y_2, triangles, firstImage, secondImage, limit, alpha):
    
    #The amount of blending is controlled by alpha
    #To be able to do face morphing, I need a morphed image
    #xm and ym holds the coordinates of morphed image 
    
    xm = []
    ym = []

    for i in range(0, 68):
        x = (1-alpha)*landmarks_x_2[i] + alpha*landmarks_x_1[i]
        y = (1-alpha)*landmarks_y_2[i] + alpha*landmarks_y_1[i]
        
        xm.append(x)
        ym.append(y)
        
 
    #Also the corners and the midpoints of the edges are needed to be saved
    landmarks_x_2.append(0)
    landmarks_y_2.append(0)
    
    landmarks_x_2.append(0)
    landmarks_y_2.append(rows/2)
    
    landmarks_x_2.append(int(cols/2))
    landmarks_y_2.append(0)
    
    landmarks_x_2.append(cols-1)
    landmarks_y_2.append(0)
    
    landmarks_x_2.append(cols-1)
    landmarks_y_2.append(rows/2)
    
    landmarks_x_2.append(0)
    landmarks_y_2.append(rows-1)
    
    landmarks_x_2.append(int(cols/2))
    landmarks_y_2.append(rows-1)
    
    landmarks_x_2.append(cols-1)
    landmarks_y_2.append(rows-1)
    
    ##################################################
    
    xm.append(0)
    ym.append(0)
    
    xm.append(0)
    ym.append(rows/2)
    
    xm.append(cols/2)
    ym.append(0)
    
    xm.append(cols-1)
    ym.append(0)
    
    xm.append(cols-1)
    ym.append(rows/2)
    
    xm.append(0)
    ym.append(rows-1)
    
    xm.append(cols/2)
    ym.append(rows-1)
    
    xm.append(cols-1)
    ym.append(rows-1)
    
    ###################################################
    
    landmarks_x_1.append(0)
    landmarks_y_1.append(0)
    
    landmarks_x_1.append(0)
    landmarks_y_1.append(rows/2)
    
    landmarks_x_1.append(int(cols/2))
    landmarks_y_1.append(0)
    
    landmarks_x_1.append(cols-1)
    landmarks_y_1.append(0)
    
    landmarks_x_1.append(cols-1)
    landmarks_y_1.append(rows/2)
    
    landmarks_x_1.append(0)
    landmarks_y_1.append(rows-1)
    
    landmarks_x_1.append(int(cols/2))
    landmarks_y_1.append(rows-1)
    
    landmarks_x_1.append(cols-1)
    landmarks_y_1.append(rows-1)
    

    #I also send triangle array to this function as a parameter. This triangles are using as reference,
    #By using these, I find vertices' ids and save these ids to be able to find correct vertices from other image
    #Thus I can receive matched triangles in both two images
    indexes_triangles = []
    index = 0
    index1 = 0
    index2 = 0
    index3 = 0
    
    #Traverse all triangles
    for i in range(len(triangles)):
        sel_triangle = triangles[i].astype(np.int)
        
        for points in sel_triangle:
            
            if (index % 6 == 0):
                for a in range(76):
                    #now landmark arrays have 68+8 points because corners are also included
                    #traverse all landmark points and try to find matched one
                    #and save the id, do this operation for all three vertices
                    if ((sel_triangle[0] == landmarks_x_2[a]) and (sel_triangle[1] == landmarks_y_2[a])):
                        index1 = a
                        break
                     
                for b in range(76):
                    
                    if ((sel_triangle[2] == landmarks_x_2[b]) and (sel_triangle[3] == landmarks_y_2[b])):
                        index2 = b
                        break
                        
                for c in range(76):
                    
                    if ((sel_triangle[4] == landmarks_x_2[c]) and (sel_triangle[5] == landmarks_y_2[c])):
                        index3 = c
                        break
                indexes_triangles.append([index1, index2, index3])      
            index = index + 1
            
     
            
    faceMorphing = np.zeros(secondImage.shape, dtype = secondImage.dtype)         

    #Ids are found now I am ready to do face morphing between matched triangles
    for i in range(len(triangles)):
        #takes the three vertices
        vertices1 = indexes_triangles[i][0]
        vertices2 = indexes_triangles[i][1]
        vertices3 = indexes_triangles[i][2]
        
        sel_triangle = triangles[i].astype(np.int)
        
        image1 = [[sel_triangle[0], sel_triangle[1]], [sel_triangle[2], sel_triangle[3]], [sel_triangle[4], sel_triangle[5]]]
        image2 = [[landmarks_x_1[vertices1], landmarks_y_1[vertices1]], [landmarks_x_1[vertices2], landmarks_y_1[vertices2]], [landmarks_x_1[vertices3], landmarks_y_1[vertices3]]]
        morhed = [[xm[vertices1], ym[vertices1]], [xm[vertices2], ym[vertices2]], [xm[vertices3], ym[vertices3]]]
        
        #Since warpAffine takes an image and not a triangle, I find bounding boxes for the triangles
        bounding1 = cv2.boundingRect(np.float32([image1]))
        bounding2 = cv2.boundingRect(np.float32([image2]))
        boundingM = cv2.boundingRect(np.float32([morhed]))
        
        row = boundingM[3]
        col = boundingM[2]
        #create a mask image to extract the desired area from image
        maskImage = np.zeros((row, col, 3), dtype = np.float32)
        
        #extract the bounding boxes from images
        image_1 = secondImage[bounding1[1]:bounding1[1] + bounding1[3], bounding1[0]:bounding1[0] + bounding1[2]]
        image_2 = firstImage[bounding2[1]:bounding2[1] + bounding2[3], bounding2[0]:bounding2[0] + bounding2[2]]
 
        
        #Saves reference points
        image1_ref = []
        image2_ref = []
        morp_ref = []
    
        
        for k in range(0, 3):
            #Takes left top vertices as reference points and saves them
            left_x_1 = image1[k][0] - bounding1[0]
            left_y_1 = image1[k][1] - bounding1[1]
            
            left_x_2 = image2[k][0] - bounding2[0]
            left_y_2 = image2[k][1] - bounding2[1]
            
            left_x_m = morhed[k][0] - boundingM[0]
            left_y_m = morhed[k][1] - boundingM[1]
            
            image1_ref.append((left_x_1,left_y_1))
            image2_ref.append((left_x_2,left_y_2))
            morp_ref.append((left_x_m,left_y_m))
        
        #fills the box
        cv2.fillConvexPoly(maskImage, np.int32(morp_ref), (1,1,1));

        
        #Apply affine transform to find transformation matrix
        transformationMatrix1 = cv2.getAffineTransform( np.float32(image1_ref), np.float32(morp_ref))
        transformationMatrix2 = cv2.getAffineTransform( np.float32(image2_ref), np.float32(morp_ref))
        
        #Apply transformation matrixes to points
        warp1 = cv2.warpAffine( image_1, transformationMatrix1, (col,row), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )        
        warp2 = cv2.warpAffine( image_2, transformationMatrix2, (col,row), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
        
        
        #Do alpha blending
        warpImage = (1.0 - alpha) * warp1 + alpha * warp2
        warpImage = warpImage * maskImage
        
        #Put the warpImage on the correct position of the image
        faceMorphing[boundingM[1]:boundingM[1]+row, boundingM[0]:boundingM[0]+col] = faceMorphing[boundingM[1]:boundingM[1]+row, boundingM[0]:boundingM[0]+col]* ( 1 - maskImage ) 
        faceMorphing[boundingM[1]:boundingM[1]+row, boundingM[0]:boundingM[0]+col] = faceMorphing[boundingM[1]:boundingM[1]+row, boundingM[0]:boundingM[0]+col] + warpImage
    
        
    return faceMorphing


#Calls function in the part3 to take image landmarks
#and calls subdivPoints function in the part3 to take triangles
landmarkPoints = landmarks(firstImage)
landmarks_x_1 = []
landmarks_y_1 = []

for i in range(0, 68):
    landmarks_x_1.append(landmarkPoints.part(i).x)
    landmarks_y_1.append(landmarkPoints.part(i).y)
    
triangles_1 = subdivPoints(firstImage, landmarks_x_1, landmarks_y_1)


#Calls function in the part3 to take image landmarks
#and calls subdivPoints function in the part3 to take triangles
landmarkPoints = landmarks(secondImage)
landmarks_x_2 = []
landmarks_y_2 = []

for i in range(0, 68):
    landmarks_x_2.append(landmarkPoints.part(i).x)
    landmarks_y_2.append(landmarkPoints.part(i).y)

triangles_2 = subdivPoints(secondImage, landmarks_x_2, landmarks_y_2)

#Calls function in the part2 to take cat landmarks
#and calls subdivPoints function in the part3 to take triangles
catLandmark_x, catLandmark_y = catLandmarks()
triangles_3 = subdivPoints(catImage, catLandmark_x, catLandmark_y)

#Saves video frames
imagesList1 = []
imagesList2 = []

imagesList3 = []
imagesList4 = []

imagesList5 = []
imagesList6 = []

#I decided to do image morphing in 50 steps
stepNumber = 50
for limit in range(0, stepNumber):
    #Each step find a ratio and send as a parameter
    t=limit/(stepNumber-1)
    
    #Sends face landmarks, second image's triangles, images and the limit to function
    img1 = morphing(landmarks_x_1, landmarks_y_1, landmarks_x_2, landmarks_y_2, triangles_2, firstImage, secondImage, limit,t)
    imagesList1.append(img1)
    
    img2= morphing(landmarks_x_2, landmarks_y_2, landmarks_x_1, landmarks_y_1, triangles_1, secondImage, firstImage, limit,t)
    imagesList2.append(img2)
    
    img3 = morphing(catLandmark_x, catLandmark_y, landmarks_x_1, landmarks_y_1, triangles_1, catImage, firstImage, limit,t)
    imagesList3.append(img3)
    
    img4 = morphing(landmarks_x_1, landmarks_y_1, catLandmark_x, catLandmark_y, triangles_3, firstImage, catImage, limit,t)
    imagesList4.append(img4)
    
    img5 = morphing(catLandmark_x, catLandmark_y, landmarks_x_2, landmarks_y_2, triangles_2, catImage, secondImage, limit,t)
    imagesList5.append(img5)
    
    img6 = morphing(landmarks_x_2, landmarks_y_2, catLandmark_x, catLandmark_y, triangles_3, secondImage, catImage, limit,t)
    imagesList6.append(img6)
    
#Since moviepy works with RGB images, I converted them into RGB
for i in range(len(imagesList1)):
    imagesList1[i] = cv2.cvtColor(imagesList1[i], cv2.COLOR_BGR2RGB)
    imagesList2[i] = cv2.cvtColor(imagesList2[i], cv2.COLOR_BGR2RGB)
    imagesList3[i] = cv2.cvtColor(imagesList3[i], cv2.COLOR_BGR2RGB)
    imagesList4[i] = cv2.cvtColor(imagesList4[i], cv2.COLOR_BGR2RGB)
    imagesList5[i] = cv2.cvtColor(imagesList5[i], cv2.COLOR_BGR2RGB)
    imagesList6[i] = cv2.cvtColor(imagesList6[i], cv2.COLOR_BGR2RGB)


#Creates videos
clip = mpy.ImageSequenceClip(imagesList1, fps=25) 
clip.write_videofile("part5_video1.mp4")

clip = mpy.ImageSequenceClip(imagesList2, fps=25) 
clip.write_videofile("part5_video2.mp4")

clip = mpy.ImageSequenceClip(imagesList3, fps=25) 
clip.write_videofile("part5_video3.mp4")

clip = mpy.ImageSequenceClip(imagesList4, fps=25) 
clip.write_videofile("part5_video4.mp4")

clip = mpy.ImageSequenceClip(imagesList5, fps=25) 
clip.write_videofile("part5_video5.mp4")

clip = mpy.ImageSequenceClip(imagesList6, fps=25) 
clip.write_videofile("part5_video6.mp4")



#Creates gifs
imagesList1 = imagesList1 + imagesList2
imagesList3 = imagesList3 + imagesList4
imagesList5 = imagesList5 + imagesList6

clip = mpy.ImageSequenceClip(sequence=imagesList1, fps=10) 
clip.write_gif('part5_gif1.gif')

clip = mpy.ImageSequenceClip(sequence=imagesList3, fps=10) 
clip.write_gif('part5_gif2.gif')

clip = mpy.ImageSequenceClip(sequence=imagesList5, fps=10) 
clip.write_gif('part5_gif3.gif')
















