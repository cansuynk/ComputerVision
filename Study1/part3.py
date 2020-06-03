import cv2
import os 
import moviepy.editor as mpy
import numpy as np

main_img_dir = './DAVIS-JPEGImages/JPEGImages/night-race/000'
main_seg_dir = './DAVIS-JPEGImages/Annotations/night-race/000'

all_images = ['00.jpg', '01.jpg', '02.jpg', '03.jpg', '04.jpg', '05.jpg', '06.jpg', '07.jpg', '08.jpg', '09.jpg']
#Keeps the images' names in order, I wrote the names which are 0 to 9 to the array manually. 

#For pictures after 10, I used for loop.
list2 = list(range(10, 46))
for i in range(len(list2)):
    all_images.append(str(list2[i])+'.jpg')
    
    
#Loads target images    
target1 = cv2.imread('./target.png') 
target2 = cv2.imread('./target2.png') 
target3 = cv2.imread('./target3.png') 

img = cv2.imread(main_img_dir +all_images[0])
R, C, B =  img.shape

#Variables for finding average histogram matching, keeps the sum of the histogram of the channels
hist_b_sum = 0
hist_g_sum = 0
hist_r_sum = 0

for i in range(len(all_images)) :
    #Reads images from the folder
    img = cv2.imread(main_img_dir +all_images[i])
    
    #Finds summation of the histogram of the blue channels
    hist_b,bins_b = np.histogram(img[:,:,0].flatten(),256,[0,256])
    hist_b_sum = hist_b_sum + hist_b
    
    #Finds summation of the histogram of the green channels
    hist_g,bins_g = np.histogram(img[:,:,1].flatten(),256,[0,256])
    hist_g_sum = hist_g_sum + hist_g
    
     #Finds summation of the histogram of the red channels
    hist_r,bins_r = np.histogram(img[:,:,2].flatten(),256,[0,256])
    hist_r_sum = hist_r_sum + hist_r
    
   
#Takes the average of the all channels
hist_b = hist_b_sum/len(all_images)*1.0
hist_g = hist_g_sum/len(all_images)*1.0
hist_r = hist_r_sum/len(all_images)*1.0

#Finds the cdf function of the histograms of the channels
#And to be able to get values between 0 to 1, it finds the normalized of the cdf
cdf_b = hist_b.cumsum()
cdf_normalized_b = cdf_b / cdf_b.max()*1.0

cdf_g = hist_g.cumsum()
cdf_normalized_g = cdf_g / cdf_g.max()*1.0

cdf_r = hist_r.cumsum()
cdf_normalized_r = cdf_r / cdf_r.max()*1.0


def histogram_matching(img, img2):
       
    #Same operations are applied: histograms of each channel, found the cdf and normalized it
    hist_t_b,bins_t_b = np.histogram(img2[:,:,0].flatten(),256,[0,256])
    cdf_t_b = hist_t_b.cumsum()
    cdf_normalized_t_b = cdf_t_b / cdf_t_b.max()*1.0

    
    hist_t_g,bins_t_g = np.histogram(img2[:,:,1].flatten(),256,[0,256])
    cdf_t_g = hist_t_g.cumsum()
    cdf_normalized_t_g = cdf_t_g / cdf_t_g.max()*1.0
  
    
    hist_t_r,bins_t_r = np.histogram(img2[:,:,2].flatten(),256,[0,256])
    cdf_t_r = hist_t_r.cumsum()
    cdf_normalized_t_r = cdf_t_r / cdf_t_r.max()*1.0

    #LUT tables for each channel
    LUT_b = np.zeros(256)
    LUT_g = np.zeros(256)
    LUT_r = np.zeros(256)
    
    gj = 0 #index
    
    
    #The process for the creating LUT tables for each channel (Taken from course slides)
    for gi in range(256):
        while cdf_normalized_t_b[gj] < cdf_normalized_b[gi] and gj < 255:
            gj = gj + 1
        LUT_b[gi] = gj
        
    gj = 0
    gi = 0
    for gi in range(256):
        while cdf_normalized_t_g[gj] < cdf_normalized_g[gi] and gj < 255:
            gj = gj + 1
        LUT_g[gi] = gj
        
    gj = 0
    gi = 0
    for gi in range(256):
        while cdf_normalized_t_r[gj] < cdf_normalized_r[gi] and gj < 255:
            gj = gj + 1
        LUT_r[gi] = gj
    
    
    #Finds proper intensity values corresponding to each pixel in the input picture from LUT
    #and writes them into the new matrix.
    K = np.zeros([R, C, B], dtype=np.uint8)
       
    
    K[:,:,0] = np.uint8(LUT_b[img[:,:,0]])
    K[:,:,1] = np.uint8(LUT_g[img[:,:,1]])
    K[:,:,2] = np.uint8(LUT_r[img[:,:,2]])


    return K


video_frames = []
 
for i in range(len (all_images)) :
    image = cv2.imread(main_img_dir +all_images[i])
    seg = cv2.imread(main_seg_dir +all_images[i].split('.')[0]+'.png ',cv2.IMREAD_GRAYSCALE)

    #histogram matching between the target image and the input image
    object1 = histogram_matching(image, target1)
    #I selected the index of the background in the image to mask. 
    #To be able to detect black color, 0 is given as index value.
    mask = cv2.inRange(seg, 0, 0)
    #To be able to take object from the input image, cv2.bitwise_and() is used
    background = cv2.bitwise_and(object1,object1, mask= mask)  #background object
    
    
    object2 = histogram_matching(image, target2)
    #To be able to detect red color, 38 is given as index value.
    mask2 = cv2.inRange(seg, 38, 38)
    image_of_the_guy = cv2.bitwise_and(object2,object2, mask= mask2)   #first guy object
    
    
    object3 = histogram_matching(image, target3)
    #To be able to detect green color, 75 is given as index value.
    mask3 = cv2.inRange(seg, 75, 75)
    image_of_the_guy2 = cv2.bitwise_and(object3,object3, mask= mask3)   #second guy object

    #Combines the three images that are applied the histogram matching and that have individual objects.
    video_frames.append(image_of_the_guy2+image_of_the_guy+background)



#To be able to create a video with the size of the images, the size info is saved.
height,width,layers=video_frames[1].shape

#Creates a video structure
video=cv2.VideoWriter('part3.mp4',-1,15,(width,height))

for j in range(len(video_frames)):
    video.write(video_frames[j])

cv2.destroyAllWindows()
video.release()
