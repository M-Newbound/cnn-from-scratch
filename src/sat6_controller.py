"""
sat6_controller.py

This script is responsible for training and testing a Convolutional Neural Network (CNN) on the SAT-6 dataset.

The SAT-6 dataset consists of satellite images labeled with one of six classes. The script loads the dataset, preprocesses it, 
defines the CNN model, trains the model on the training data, and tests the model on the test data. It also includes functionality 
for saving and loading the trained model, and for making a single prediction on an individual image.
"""


#%% 
# preprocess data

import numpy as np
from scipy.io import loadmat


data = loadmat('./data/sat-6-full.mat')

train_x = data['train_x']
train_y = data['train_y'].T

test_x = data['test_x']
test_y = data['test_y'].T

train_x = np.transpose(train_x, (3, 1, 0, 2))
test_x = np.transpose(test_x, (3, 1, 0, 2))


#%% 
# Define model topology

from src.model.cnn import ConvolutionalNeuralNetwork
from src.model.dense import Dense
from src.model.conv2d import Conv2D
from src.model.maxpool2d import MaxPool2D
from src.model.flattern import Flattern

layers = [
    Conv2D(8, 3, 'relu'),
    MaxPool2D(2),
    Flattern(),
    Dense(1280, 'sigmoid'),
    Dense(640, 'relu'),
    Dense(6, 'sigmoid')
]

cnn = ConvolutionalNeuralNetwork(layers)


#%% 
# Save the model
cnn.save('./models/sat6_model_v1.pkl')

#%% 
# Load the model
from src.model.cnn import ConvolutionalNeuralNetwork

cnn = ConvolutionalNeuralNetwork.load('./models/sat6_model_v1.pkl')

#%% 
# Train the model
learning_rate = 0.0001
decay_rate = 0.99
epochs = 1000

batch_size = 75
num_batches = min(len(train_x) // batch_size, 10)

offset = 0

for epoch in range(epochs):
    total_loss = 0

    for i in range(num_batches):
        start = (i + offset) * batch_size
        end = start + batch_size
        train_images = train_x[start:end]
        train_labels = train_y[start:end]
        loss = cnn.train(train_images, train_labels, learning_rate, 'binary_crossentropy')
        total_loss += loss
        # print(f'Epoch {epoch + 1} Batch {i + 1} Loss: {loss}')
    
    print(f'Epoch {epoch + 1} Avg Loss: {total_loss/num_batches} Learning Rate: {learning_rate}')
    if epoch % 3 == 0 : offset += 1

#%% 
# test the model
from sklearn.metrics import accuracy_score

pred_x = test_x[:750]
pred_y = test_y[:750]

predictions = cnn.predict(pred_x)
predictions = np.argmax(predictions, axis=1)
true_labels = np.argmax(pred_y, axis=1)

accuracy = accuracy_score(true_labels, predictions)
print(f'Accuracy: {accuracy}')

#%% 
# Singluar prediction with image shown

import matplotlib.pyplot as plt

index = 8
image = test_x[index]
label = test_y[index]

prediction = cnn.predict(image.reshape(-1, 28, 28, 4))
prediction = np.argmax(prediction)

plt.title(f'Predicted: {prediction}, True: {np.argmax(label)}')

plt.imshow(test_x[index])
plt.show()

# %%
