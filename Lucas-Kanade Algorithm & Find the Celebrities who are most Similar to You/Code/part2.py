# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
from sklearn import neighbors
import cv2
import os
import os.path
import pickle
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

#Function apply dimension reduction and convert it to 1d vector
def image_to_vector(image):
    
    image = PCA(n_components=3).fit_transform(image)
    size = image.shape
    image_vector = cv2.resize(image, size).flatten()
    return image_vector

#Function traverses all folders and reads all images inside folders
#Prepare train data and group labels
def _prepareData(train_data_dir):
    print("Data is preparing...")
    
    X = []
    y = []
    
    for class_dir in os.listdir(train_data_dir):
        if not os.path.isdir(os.path.join(train_data_dir, class_dir)):
            continue

        for image_path in os.listdir(os.path.join(train_data_dir, class_dir)):
            path = os.path.join(train_data_dir, class_dir)           
            
            image = cv2.imread(os.path.join(path, image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_vector = image_to_vector(image)
            
            X.append(image_vector)
            y.append(class_dir)
            
    return X, y

#Train data by using k nearest neighbors
def _train(X, y, model_save_path=None, trainAlgo='auto'):
    print("Model is preparing...")
    
    model = neighbors.KNeighborsClassifier(n_neighbors=10, algorithm=trainAlgo, weights='distance')
    model.fit(X, y)
    
    #Model can be saved for reusing
    if model_save_path is not None:
        with open(model_save_path, 'wb') as file:
            pickle.dump(model, file)
    
    return model

    
#Predict an input image 
def _predict(image_path, model=None, model_save_path=None):
    print("Prediction is doing...")
    
    #If model is saved before, it can be taken 
    if model is None and model_save_path is None:
        raise Exception("There is no model or model path")

    if model is None:
        with open(model_save_path, 'rb') as file:
            model = pickle.load(file)
    
    #Convert image to gray scale and apply dimension reduction
    my_image = cv2.imread(image_path)
    my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)
    my_image_vector = image_to_vector(my_image)   
    my_image_vector = my_image_vector.reshape(1, -1)
    
    predict = model.predict(my_image_vector)
    return predict
    
#I used this method to show result as an image
#It takes first image of the result person from its folder
def _show_who_are_you(my_image_path, main_dir, result):
    
    out_path = os.path.join(main_dir, result[0])
    out_path =  os.path.join(out_path, os.listdir(os.path.join(main_dir, result[0]))[0])
    
    input_image = cv2.imread(my_image_path)
    output_image = cv2.imread(out_path)
    
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    
    #Makes bigger of images
    input_image = cv2.resize(input_image,(150,200))
    output_image = cv2.resize(output_image,(150,200))
    
    horizontal_concat = np.concatenate((input_image, output_image), axis=1)
    
    cv2.imshow('Who are you?: ' + result[0], horizontal_concat)
    cv2.waitKey()
    

main_dir = './VGGFace-subset'
my_image_path = './cansu.jpg'
#my_image_path = './cansu2.jpg'

X, y = _prepareData(main_dir)

model = _train(X, y)
result = _predict(my_image_path, model)
print("Result is: ",result[0])

_show_who_are_you(my_image_path, main_dir, result)


'''
#The KNN model can be saved after training and load it to make prediction
model_save_path = './SaveModel.clf'
model = _train(X, y, model_save_path)
result = _predict(my_image_path, None, model_save_path)
print("Result is: ",result[0])
_show_who_are_you(my_image_path, main_dir, result)
'''




                        
                        
                        