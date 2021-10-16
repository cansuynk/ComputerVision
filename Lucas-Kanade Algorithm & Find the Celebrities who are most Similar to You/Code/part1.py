# -*- coding: utf-8 -*-


import moviepy.video.io.VideoFileClip as mpy
import moviepy.editor as mp
import cv2
import numpy as np
import scipy.signal as signal
import numpy.linalg as lin

walker = mpy.VideoFileClip("walker.avi")
walker_hand = mpy.VideoFileClip("walker_hand.avi")

frame_count = walker_hand.reader.nframes
video_fps = walker_hand.fps


walker_frames = []
walker_hand_frames = []

for i in range(frame_count):
    walker_frame = walker.get_frame(i*1.0/video_fps)
    walker_hand_frame = walker_hand.get_frame(i*1.0/video_fps)
    
    #walker_hand_frame = (walker_hand_frame > 127)
    
    if(i%2==0):
        walker_frames.append(walker_frame)
        walker_hand_frames.append(walker_hand_frame)
        

images = []
hand_area = []
hand_area_top = []
hand_area_bottom = []

for i in range(len(walker_frames)):
    
    x = walker_frames[i]
    y = walker_hand_frames[i]
    
    #Apply Canny to hand image to detect hand edges 
    gray = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(gray,(5,5),0)
    max_treshold,img = cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_treshold = max_treshold/3
    image = cv2.Canny(image,low_treshold,max_treshold)
    
    #Apply contour
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    image = cv2.drawContours(image, contours, -1, (255,255,255), 1)
    
    #Find white pixels
    b, a = np.where(image == 255)
    
    #Create a bounding box that contains hand
    topy, topx = (np.min(b), np.min(a))
    bottomy, bottomx = (np.max(b), np.max(a))
    
    #To make bigger of the box area
    bottomx = topx+50
    bottomy = topy+50
    
    topx = topx-50
    topy = topy-50
    
    hand_area_top.append([topx,topy])
    hand_area_bottom.append([bottomx,bottomy])
    
    #Extract hand area from the image and saves it
    hand = x[topy:bottomy+1, topx:bottomx+1]
    hand_area.append(hand)
    
    #Draw area to original frame image
    img = cv2.rectangle(img=x, pt1=(topx, topy), pt2=(bottomx, bottomy),
                 color=(0, 0, 255),
                 thickness=1)
    
    #Saves frame
    images.append(img);



def detectMotion(image1, image2, windowSize):
    
    threshold=0.003
    mode = 'same'
    boundary = 'symm'
    u = np.zeros(image1.shape)
    v = np.zeros(image1.shape)
    
    #Kernels to find gradients (derivatives)
    Kx = np.array([[-1., 1.], [-1., 1.]])
    Ky = np.array([[-1., -1.], [1., 1.]])
    Kt = np.array([[1., 1.], [1., 1.]])
   
    w = int(windowSize/2)
    
    #normalize pixels
    image1 = image1 / 255.
    image2 = image2 / 255.
    
    #Find derivatives
    fx = signal.convolve2d(image1, Kx, boundary=boundary, mode=mode)
    fy = signal.convolve2d(image1, Ky, boundary=boundary, mode=mode)
    ft = signal.convolve2d(image2, Kt, boundary=boundary, mode=mode) + signal.convolve2d(image1, -Kt, boundary=boundary, mode=mode)

    # within window window_size * window_size
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            #Find b
            b = np.reshape(It, (It.shape[0],1))
           
            A = np.vstack((Ix, Iy)).T   #A = [Ix Iy]
            X = np.matmul(A.T, A)
            # the eigenvalues of A.TA should not be too small
            if np.min(abs(np.linalg.eigvals(X))) >= threshold:
                X = np.linalg.pinv(X)
                X = -1 * X
                B = np.matmul(A.T, b)
                result = np.matmul(X, B)
                u[i,j]=result[0]
                v[i,j]=result[1]          
 
    return (u,v)


def drawArrows(image, flag, u=None, v=None, flowVector=None):

    step=12
    row, col, h = image.shape
    
    #Separate area into steps to show arrows
    y, x = np.mgrid[step/2:row:step, step/2:col:step].reshape(2,-1).astype(int)
    
    if (flag==1):
        fx, fy = flowVector[y,x].T
        fx = fx*2
        fy = fy*2
    else:
        fx = u[y,x].T
        fy = v[y,x].T
    
    #Find starting and end points of arrow
    arrows = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    arrows = np.int32(arrows + 0.5)

    color = (0, 255, 0)
    thickness = 2 
    length = 0.5
    
    #Draw arrows
    for (x1, y1), (x2, y2) in arrows:
        image = cv2.arrowedLine(image, (x1, y1), (x2, y2), color, thickness, tipLength = length)
    return image



frames = []
j = 0
for i in range(len(walker_frames)-1):
    
    print("Step: ",i+1)
    j = i + 1
    
    image = images[i]
    image1c = hand_area[i]
    image2c = hand_area[j]
    
    #Convert consecutive images to gray scale
    image1 = cv2.cvtColor(image1c, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2c, cv2.COLOR_BGR2GRAY)
    
    #Image1 bounding box coordinates
    topx_1 = hand_area_top[i][0]
    topy_1 = hand_area_top[i][1]
    
    bottomx_1 = hand_area_bottom[i][0]
    bottomy_1 = hand_area_bottom[i][1]
    
    
    #Find the motion field for only hand area
    u1, v1 = detectMotion(image1, image2, 15)
    out = drawArrows(image1c, 0, u1, v1, None)
    
    #Put result on original image
    image = images[i].copy()
    image[topy_1:bottomy_1+1, topx_1:bottomx_1+1] = out
    frames.append(image)
   
    
    '''
    #Here, I used existing function to see proper result, I also uploaded this result
    flowVector = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    out = drawArrows(image1c, 1, None, None, flowVector)
    
    image = images[i].copy()
    image[topy_1:bottomy_1+1, topx_1:bottomx_1+1] = out
    frames.append(image)
    '''

#I obtain a slower video
clip = mp.ImageSequenceClip(frames, fps=10) 
clip.write_videofile("part1_myResult.mp4")






