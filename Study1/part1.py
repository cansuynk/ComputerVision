import cv2
import os 
import moviepy.editor as mpy
import numpy as np

main_img_dir = './DAVIS-JPEGImages/JPEGImages/walking/000'
main_seg_dir = './DAVIS-JPEGImages/Annotations/walking/000'

all_images = ['00.jpg', '01.jpg', '02.jpg', '03.jpg', '04.jpg', '05.jpg', '06.jpg', '07.jpg', '08.jpg', '09.jpg']
#Keeps the images' names in order, I wrote the names which are 0 to 9 to the array manually. 

#For pictures after 10, I used for loop. 
list2 = list(range(10, 72))
for i in range(len(list2)):
    all_images.append(str(list2[i])+'.jpg')


images_list = [] #Keeps all masked image frames in order to use for the video

#This for loop reads each image and the segmentation maps that belong to that image, 
#and using the segmentation maps, masks some color channels of the images.
for i in range(len (all_images)) :
    image = cv2.imread(main_img_dir +all_images[i])
	# Image is a numpy array with shape(800 , 1920 , 3)
    seg = cv2.imread(main_seg_dir +all_images[i].split('.')[0]+'.png ',cv2.IMREAD_GRAYSCALE)
	#Seg is the segmentation map of the image with shape(800 , 1920)

	#I selected the index of the guy in the image to mask. To be able to detect red color, 38 is given as index value.
    mask = cv2.inRange(seg, 38, 38)	
	#To be able to mask the out of the guy
    mask2 = cv2.bitwise_not(mask)
    
	#To be able to take object from the input image, cv2.bitwise_and() is used
    image_without_the_guy = cv2.bitwise_and(image,image, mask= mask2)
    image_of_the_guy = cv2.bitwise_and(image,image, mask= mask)
    image_of_the_guy[:,:,1:3]=image_of_the_guy[:,:,1:3]*25/100
    #I decreased the values of red and green channels by 75%.
    
	#Combines the extracted object and the area outside the object as an output image.
    image = (image_without_the_guy + image_of_the_guy)
	#Saves the masked image into array
    images_list.append(image)

#To be able to create a video with the size of the images, the size info is saved.
height,width,layers=images_list[1].shape

#Creates a video structure

video=cv2.VideoWriter('part1.mp4',-1,15,(width,height))

for j in range(len(images_list)):
    video.write(images_list[j])

cv2.destroyAllWindows()
video.release()


'''
#To display an image to the screen
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.imshow('res',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
