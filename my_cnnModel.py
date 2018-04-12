from keras.models import Sequential 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#dense for fully connectd layer creation

#initalising the neural network model
classifier=Sequential()
#adding the convolutionl layer 
#using tensorflow backend
classifier.add(Conv2D(32, (3, 3), input_shape=(128,128, 3), activation="relu"))
#adding the maxpooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(32, (3, 3), activation="relu"))
#adding the maxpooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding the flattening layer
classifier.add(Flatten())

#adding fully connected layer 
#hidden layer over here of our ann
classifier.add(Dense(activation="relu",units=128))

classifier.add(Dense(activation="relu",units=64))
classifier.add(Dense(activation="relu",units=32))
classifier.add(Dense(activation="relu",units=32))
#adding output layer
#here we are having five classes
classifier.add(Dense(activation='softmax',units=10))

#classifier

#compiling the neural net
#adam is stocatic gradient descent algo
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

'''
#image preprocessing step
#image augmentation step
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128,128),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
					        'dataset/test_set',
					        target_size=(128,128),
					        batch_size=32,
					        class_mode='categorical')
'''
'''
classifier.fit_generator(
        training_set,
        steps_per_epoch=700,
        epochs=25,
        validation_data=test_set,
        validation_steps=150)

classifier.save('mymodel.h5')
classifier.save_weights('mymodel_weights.h5')
'''
'''
import cv2 
import numpy as np

img=cv2.imread('dataset/single_prediction/cat_or_dog_1.jpg')
img1=cv2.imread('dataset/single_prediction/cat_or_dog_2.jpg')

img=cv2.resize(img,(32,32))
img=np.reshape(img,[1,32,32,3])
img1=cv2.resize(img1,(32,32))
img1=np.reshape(img1,[1,32,32,3])
#here 1st argument coreponds to batch
#2,3,4 arguments include image height,width,number of channels
classes=classifier.predict(img)
classes1=classifier.predict(img1)


#for single prediction(testing)
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
'''