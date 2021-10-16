# -*- coding: utf-8 -*-

import cv2
import dlib

def putLandmarks(image):
    
    detector = dlib.get_frontal_face_detector()
    #Used to detect face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    #Predit landmark points

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #Predictor works only grayscale images

    rectangles = detector(gray)  #Finds list of rectangles

    #To be able to draw a rectangle to show face, I need to find rectangle coordinates
    #I used below methods
    x = rectangles[0].left()
    y = rectangles[0].top()
    w = rectangles[0].right() - x
    h = rectangles[0].bottom() - y
    
    #Draws a rectangle to show the face 
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    
    #Create points structure to store 68 landmarks
    points = predictor(gray, rectangles[0])
    
    #Draws all 68 landmarks on the faces
    for i in range(0, 68):
        x = points.part(i).x
        y = points.part(i).y 
    
        image = cv2.rectangle(image, (x, y), (x + 5, y + 5), (0, 255, 0), -1)
        
    return image
        

#Loads cat image and the points
catImage = cv2.imread("./CAT_00/00000095_001.jpg")

file = open("./CAT_00/00000095_001.jpg.cat")
content = file.read()
file.close()

content = content.split()               #parse according to space
content = [int(i) for i in content]     #convert string values to integer

#Places points on the cat image
index = 1
for i in range (0, content[0]):
    x = content[index]
    y = content[index+1]
    index = index + 2
    catImage = cv2.rectangle(catImage, (x, y), (x + 5, y + 5), (0, 255, 0), -1)
    
    
#Loads images and calls the function to put face landmarks on them
firstImage = cv2.imread("./dennis_ritchie.jpg")
secondImage = cv2.imread("./yusuf.jpg")
rows, cols, ch = catImage.shape

#Since it is necessary for all three photos to be same size, I resized the photos
firstImage = cv2.resize(firstImage,(cols,rows))
secondImage = cv2.resize(secondImage,(cols,rows))

firstImage = putLandmarks(firstImage)
secondImage = putLandmarks(secondImage)

#To save the images you can open the comments
"""
cv2.imwrite("Part1_cat.jpg", catImage)
cv2.imwrite("Part1_dennis.jpg", firstImage)
cv2.imwrite("Part1_yusuf.jpg", secondImage)
"""

#To display the images
cv2.imshow("Output1", firstImage)
cv2.imshow("Output2", secondImage)
cv2.imshow("Output3", catImage)
cv2.waitKey(0)
