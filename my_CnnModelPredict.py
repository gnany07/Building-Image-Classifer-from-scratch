#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 21:40:23 2018

@author: gnani
"""

from keras.models import load_model
import numpy as np
from keras.preprocessing import image

classifier=load_model('mymodel.h5')

test_image1 = image.load_img('dataset/single_prediction/elephant.jpg', target_size = (128, 128))
test_image2 = image.load_img('dataset/single_prediction/deer.jpg', target_size = (128, 128))
test_image3 = image.load_img('dataset/single_prediction/giraffe.jpg', target_size = (128,128))
test_image1 = image.img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1, axis = 0)
result1 = classifier.predict(test_image1)

test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis = 0)
result2 = classifier.predict(test_image2)

test_image3 = image.img_to_array(test_image3)
test_image3 = np.expand_dims(test_image3, axis = 0)
result3 = classifier.predict(test_image3)
#training_set.class_indices
f=open('train_labels.txt','r')
training_labels=f.read()
print(training_labels)
f.close()

print(result1)
print(result2)
print(result3)
y=result1.shape[1]
'''
for i in range(y):
    if result1[0][i]==1:
        for animal,number in training_labels.items():
            if number==i:
                print("the ouput of the prediction is "+animal)
'''
