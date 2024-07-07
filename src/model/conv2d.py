import numpy as np
from scipy.signal import correlate2d as conv2d
from .cnn_math import calc_activation_func, calc_activation_func_derivative

class Conv2D():
    """
    A class representing a 2D convolutional layer in a Convolutional Neural Network.
    """

    def __init__(self, num_filters, filter_size, activation_func_type):
        """
        Initialize the Conv2D layer with the given number of filters, filter size, and activation function type.

        Parameters:
        num_filters: int
            The number of filters in the convolutional layer.
        filter_size: int
            The size of the filters in the convolutional layer.
        activation_func_type: str
            The type of activation function to use ('relu' or 'sigmoid' supported).
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.activation_func_type = activation_func_type

        self.filters = None
        self.d_L_d_filters = None


    def forward(self, input):
        """
        Perform forward propagation through the convolutional layer.

        Parameters:
        input: numpy array of shape (batch_size, h, w, num_input_channels)
            The input data for the batch.

        Returns:
        output: numpy array of shape (batch_size, h - filter_size + 1, w - filter_size + 1, num_filters)
            The output from the convolutional layer.
        """
        self.last_input = input

        batch_size, h, w, self.num_input_channels = input.shape
        if self.filters is None:
            self.filters = np.random.randn(self.num_filters, self.num_input_channels, self.filter_size, self.filter_size) / (self.filter_size * self.filter_size)
    
        output = np.zeros((batch_size, h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))

        for b in range(batch_size):
            for i in range(self.num_filters):
                for j in range(self.num_input_channels):
                    output[b, :, :, i] += conv2d(input[b, :, :, j], self.filters[i, j], mode='valid')

        return calc_activation_func(output, self.activation_func_type)


    def backward(self, d_L_d_out):
        """
        Perform backward propagation through the convolutional layer.

        Parameters:
        d_L_d_out: numpy array of shape (batch_size, h, w, num_filters)
            The gradient of the loss with respect to the output from the convolutional layer.

        Returns:
        d_L_d_input: numpy array of shape (batch_size, h, w, num_input_channels)
            The gradient of the loss with respect to the input to the convolutional layer.
        """
        batch_size, _, _, _ = d_L_d_out.shape
        self.d_L_d_filters = np.zeros(self.filters.shape)
        d_L_d_input = np.zeros(self.last_input.shape)

        for b in range(batch_size):
            for i in range(self.num_filters):
                for j in range(self.num_input_channels):
                    self.d_L_d_filters[i, j] += conv2d(self.last_input[b, :, :, j], d_L_d_out[b, :, :, i], mode='valid')
                    d_L_d_input[b, :, :, j] += conv2d(d_L_d_out[b, :, :, i], np.rot90(self.filters[i, j], 2), mode='full')

        return d_L_d_input


    def update(self, learning_rate):
        """
        Update the parameters of the convolutional layer using the given learning rate.

        Parameters:
        learning_rate: float
            The learning rate to use for parameter updates.
        """
        self.filters -= learning_rate * self.d_L_d_filters