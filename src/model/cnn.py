"""
cnn.py

This module defines a Convolutional Neural Network (CNN) class, 
including methods for forward and backward propagation, 
training, prediction, and saving/loading the model.
"""
from .cnn_math import calc_loss, calc_loss_gradient
import pickle
import os

class ConvolutionalNeuralNetwork():
    """
    A class representing a Convolutional Neural Network.
    """

    def __init__(self, layers):
        """
        Initialize the CNN with the given layers.

        Parameters:
        layers: list of Layer objects
            The layers to include in the network.
        """
        self.layers = layers


    def _forward(self, x):
        """
        Perform forward propagation through all layers of the network.

        Parameters:
        x: numpy array of shape (batch_size, input_dim)
            The input data for the batch.

        Returns:
        x: numpy array of shape (batch_size, output_dim)
            The output from the last layer of the network.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x


    def _backward(self, loss_gradient):
        """
        Perform backward propagation through all layers of the network, in reverse order.

        Parameters:
        loss_gradient: numpy array of shape (batch_size, output_dim)
            The gradient of the loss with respect to the output from the last layer of the network.
        """
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)


    def _update(self, learning_rate):
        """
        Update the parameters of all layers using the given learning rate.

        Parameters:
        learning_rate: float
            The learning rate to use for parameter updates.
        """
        for layer in self.layers:
            layer.update(learning_rate)


    def train(self, x, y, learning_rate, loss_type):
        """
        Train the network on a batch of data.

        Parameters:
        x: numpy array of shape (batch_size, input_dim)
            The input data for the batch.
        y: numpy array of shape (batch_size, output_dim)
            The true labels for the batch.
        learning_rate: float
            The learning rate to use for parameter updates.
        loss_type: str
            The type of loss function to use ('binary_crossentropy' supported).

        Returns:
        loss: float
            The loss of the network on this batch.
        """
        y_pred = self._forward(x)
        loss = calc_loss(y_pred, y, loss_type)
        loss_gradient = calc_loss_gradient(y_pred, y, loss_type)
        self._backward(loss_gradient)
        self._update(learning_rate)
        return loss


    def predict(self, x):
        """
        Predict the output for a batch of data.

        Parameters:
        x: numpy array of shape (batch_size, input_dim)
            The input data for the batch.

        Returns:
        y_pred: numpy array of shape (batch_size, output_dim)
            The network's predictions for the batch.
        """
        return self._forward(x)
    

    def save(self, filename):
        """
        Save the model to a file.

        Parameters:
        filename: str
            The path to the file where the model should be saved.

        This method uses Python's built-in pickle module to serialize the model.
        """
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, filename):
        """
        Load a model from a file.

        Parameters:
        filename: str
            The path to the file from which the model should be loaded.

        Returns:
        model: ConvolutionalNeuralNetwork
            The loaded model.

        This method uses Python's built-in pickle module to deserialize the model.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)
