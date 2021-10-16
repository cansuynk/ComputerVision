# -*- coding: utf-8 -*-

import cv2
import numpy as np
import dlib
from part2 import catLandmarks

#Loads images
catImage = cv2.imread("./CAT_00/00000095_001.jpg")
rows, cols, ch = catImage.shape

firstImage = cv2.imread("./dennis_ritchie.jpg")
secondImage = cv2.imread("./yusuf.jpg")

#Since it is necessary for all three photos to be same size, I resized the photos
firstImage = cv2.resize(firstImage,(cols,rows))
secondImage = cv2.resize(secondImage,(cols,rows))


#This function finds the face landmarks for images
def landmarks(image):
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rectangles = detector(gray)
    
    points = predictor(gray, rectangles[0])
    
    return points

def subdivPoints (image, landmarks_x, landmarks_y):

    #Performs Delaunay Triangulation
    subdiv = cv2.Subdiv2D((0,0,image.shape[1]+1, image.shape[0]+1))
    
    #Insert landmark points
    for i in range(0, 68):
        subdiv.insert((landmarks_x[i],landmarks_y[i]))


    rows, cols, ch = image.shape
    
    #Also insert corners and the midpoints of the edges
    subdiv.insert((0,0))
    subdiv.insert((0, rows/2))
    subdiv.insert((cols/2, 0))
    subdiv.insert((cols-1, 0))
    subdiv.insert((cols-1, rows/2))    
    subdiv.insert((0, rows-1))
    subdiv.insert((cols/2, rows-1))
    subdiv.insert((cols-1, rows-1))
    
    #Obtains full list of triangles
    triangles = subdiv.getTriangleList()
    
    return triangles

#Draw triangles
def drawLines (triangles, image):
    for i in range(len(triangles)):
        sel_triangle = triangles[i].astype(np.int)
        
        
        for points in sel_triangle:
            point1 = (sel_triangle[0], sel_triangle[1])
            point2 = (sel_triangle[2], sel_triangle[3])
            point3 = (sel_triangle[4], sel_triangle[5])
            
            cv2.line(image, point1, point2, (0, 255, 0), 1)
            cv2.line(image, point2, point3, (0, 255, 0), 1)
            cv2.line(image, point1, point3, (0, 255, 0), 1)
        

################################################################################

landmarkPoints = landmarks(firstImage)
landmarks_x = []
landmarks_y = []

#I save the landmark points x and y coordinates separately
for i in range(0, 68):
    landmarks_x.append(landmarkPoints.part(i).x)
    landmarks_y.append(landmarkPoints.part(i).y)

#Find and draw triangles     
triangles_1 = subdivPoints(firstImage, landmarks_x, landmarks_y)
drawLines(triangles_1, firstImage)


landmarkPoints = landmarks(secondImage)
landmarks_x = []
landmarks_y = []

for i in range(0, 68):
    landmarks_x.append(landmarkPoints.part(i).x)
    landmarks_y.append(landmarkPoints.part(i).y)

triangles_2 = subdivPoints(secondImage, landmarks_x, landmarks_y)
drawLines(triangles_2, secondImage)


#Calls function from part2 to take landmark points of cat
catLandmark_x, catLandmark_y = catLandmarks()

triangles_3 = subdivPoints(catImage, catLandmark_x, catLandmark_y)
drawLines(triangles_3, catImage)

#To display the images you can open the comments
"""
cv2.imshow("Output1", firstImage)
cv2.imshow("Output2", secondImage)
cv2.imshow("Cat", catImage)
cv2.waitKey(0)
"""

#To save the image you can open the comments
"""        
cv2.imwrite("Part3_dennis.jpg", firstImage)
cv2.imwrite("Part3_yusuf.jpg", secondImage)
cv2.imwrite("Part3_cat.jpg", catImage)
"""     
        
        
        
        
        
        
    