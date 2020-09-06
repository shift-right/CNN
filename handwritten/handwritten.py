import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import np_utils, plot_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd

# Mnist Dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
x_train = X_train.reshape(60000, 28, 28,1)/255
x_test = X_test.reshape(10000, 28, 28,1)/255
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)

# Model Structure
model = Sequential()
# Create CN layer 1  
model.add(Conv2D(filters=16,  kernel_size=(5,5),  padding='same',  input_shape=(28,28,1),  activation='relu')) 
# Create Max-Pool 1  
model.add(MaxPooling2D(pool_size=(2,2)))  
# Create CN layer 2  
model.add(Conv2D(filters=36,  kernel_size=(5,5),  padding='same',  input_shape=(28,28,1),  activation='relu'))
# Create Max-Pool 2  
model.add(MaxPooling2D(pool_size=(2,2)))



# Add Dropout layer  
model.add(Dropout(0.25)) 
model.add(Flatten())
model.add(Dense(128, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(10, activation='softmax'))
print(model.summary())

# Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_split=0.2,epochs=10, batch_size=64, verbose=1)
# Test
loss, accuracy = model.evaluate(x_test, y_test)
print('Test:')
print('Loss: %s\nAccuracy: %s' % (loss, accuracy))

# Save model
#model.save('./CNN_Mnist.h5')

# Load Model
model = load_model('./CNN_Mnist.h5')

# Display
def plot_images_labels_prediction(images,labels,
                                  prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title= "label=" +str(labels[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx]) 
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()
plot_images_labels_prediction(X_train, Y_train,[],0,10)
