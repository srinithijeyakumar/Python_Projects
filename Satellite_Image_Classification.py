#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import os


# In[3]:


import io
import os
import matplotlib.pyplot as plt
import tensorflow as tf


# In[4]:


from matplotlib import image as mpimg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from sklearn.metrics import confusion_matrix , classification_report, ConfusionMatrixDisplay
import seaborn

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[5]:


#https://www.kaggle.com/code/kaustubhshahade/satelite-data-classification-tensorflow-cnn


# In[6]:


data_dir =r"C:\Users\Srinithi\OneDrive - Kumaraguru College of Technology\Documents\Satellite_data"
labels = os.listdir(data_dir)
labels


# In[7]:


for label in labels:
    print(label, len(os.listdir(data_dir+'/'+label)))


# In[8]:


for label in labels:
    path = os.listdir(data_dir + '/' + label)
    img = data_dir + '/' + label + '/' + path[1]
    plt.title(label)
    plt.xlabel("X pixel scaling")
    plt.ylabel("Y pixels scaling")
    image = mpimg.imread(img)
    plt.imshow(image)
    plt.show()


# In[9]:


datagen = ImageDataGenerator(
        samplewise_center=True,  # set each sample mean to 0
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,
        rescale=1./255,#scale images
        validation_split=0.2) #split data

#create training set from folders
train_data=datagen.flow_from_directory(data_dir,
                                       target_size=(64,64),
                                       batch_size=32,
                                       class_mode='categorical',
                                       shuffle=True,subset='training')

#create test set
test_data=datagen.flow_from_directory(data_dir,
                                       target_size=(64,64),
                                       batch_size=1,
                                       shuffle=False,subset='validation')


# In[10]:


img_iter = datagen.flow_from_directory(data_dir,
                                       target_size=(64,64),
                                       batch_size=32)
x, y = img_iter.next()
fig, ax = plt.subplots(nrows=4, ncols=8)
for i in range(32):
    image = x[i]
    ax.flatten()[i].imshow(np.squeeze(image))
plt.show()


# In[11]:


model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", 
                 input_shape=(64,64, 3)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=4, activation="softmax"))


# In[12]:


model.compile(loss='categorical_crossentropy', metrics=['accuracy'])


# In[13]:


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

history = model.fit(train_data,
                    validation_data=test_data,
                    epochs=20, 
                    callbacks=[callback])


# In[14]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(16, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[15]:


score = model.evaluate(test_data)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[17]:


predict=model.predict(test_data)
# predict the class label
y_classes = predict.argmax(axis=-1)
y_classes


# In[18]:


print(classification_report(test_data.classes, y_classes))


# In[ ]:




