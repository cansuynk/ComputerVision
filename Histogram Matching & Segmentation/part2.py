import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


main_img_dir = './DAVIS-JPEGImages/JPEGImages/car-turn/000'
main_seg_dir = './DAVIS-JPEGImages/Annotations/car-turn/000'

all_images = ['00.jpg', '01.jpg', '02.jpg', '03.jpg', '04.jpg', '05.jpg', '06.jpg', '07.jpg', '08.jpg', '09.jpg']
#Keeps the images' names in order, I wrote the names which are 0 to 9 to the array manually. 

#For pictures after 10, I used for loop.
list2 = list(range(10, 80))
for i in range(len(list2)):
    all_images.append(str(list2[i])+'.jpg')



img = cv2.imread(main_img_dir +all_images[0]) # load an input image with the original color on it
img2 = cv2.imread('./target.png') # load the target image


#Keeps Row, Coloumn and Bin information of the image and the target
R, C, B =  img.shape
R_t, C_t, B_t =  img2.shape


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

'''
#For plotting the cdf graphs
plt.plot(cdf_normalized_b, color = 'b')
plt.plot(cdf_normalized_g, color = 'g')
plt.plot(cdf_normalized_r, color = 'r')
'''


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

'''
#For plotting the cdf graphs
plt.plot(cdf_normalized_t_b, color = 'c')
plt.plot(cdf_normalized_t_g, color = 'y')
plt.plot(cdf_normalized_t_r, color = 'm')
'''
#LUT tables for each channel
LUT_b = np.zeros(256)
LUT_g = np.zeros(256)
LUT_r = np.zeros(256)

#index
gj = 0

#The process for the creating LUT tables for each channel (Taken from course slides)
for gi in range(256):
    while cdf_normalized_t_b[gj] < cdf_normalized_b[gi] and gj < 255:
        gj = gj + 1
    LUT_b[gi] = gj
    
gj = 0 #indexes
gi = 0
for gi in range(256):
    while cdf_normalized_t_g[gj] < cdf_normalized_g[gi] and gj < 255:
        gj = gj + 1
    LUT_g[gi] = gj
    
gj = 0 #indexes
gi = 0
for gi in range(256):
    while cdf_normalized_t_r[gj] < cdf_normalized_r[gi] and gj < 255:
        gj = gj + 1
    LUT_r[gi] = gj


#Keeps all image frames which are outputs of the histogram matching
video_frames = []

#Finds proper intensity values corresponding to each pixel in the input picture from LUT
#and writes them into the new matrix. 

for i in range(len(all_images)):
    K = np.zeros([R, C, B], dtype=np.uint8)
    img = cv2.imread(main_img_dir +all_images[i])

    K[:,:,0] = np.uint8(LUT_b[img[:,:,0]])
    K[:,:,1] = np.uint8(LUT_g[img[:,:,1]])
    K[:,:,2] = np.uint8(LUT_r[img[:,:,2]])
    
    video_frames.append(K)



#To be able to create a video with the size of the images, the size info is saved.
height,width,layers=video_frames[1].shape

#Creates a video structure
video=cv2.VideoWriter('part2.mp4',-1,15,(width,height))

for j in range(len(video_frames)):
    video.write(video_frames[j])

cv2.destroyAllWindows()
video.release()
