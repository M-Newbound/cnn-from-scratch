import numpy as np
from .cnn_math import calc_activation_func, calc_activation_func_derivative

class Dense():
    """
    Represents a fully connected layer in a Neural Network.
    """

    def __init__(self, node_count, activation_func_type):
        """
        Initializes the Dense layer.

        Parameters:
        node_count: int - The number of nodes in the layer.
        activation_func_type: str - The type of activation function.
        """
        self.activation_func_type = activation_func_type
        self.node_count = node_count

        self.weights = None
        self.bias = None


    def forward(self, X):
        """
        Performs forward propagation through the layer.

        Parameters:
        X: numpy array - The input data.

        Returns:
        numpy array - The output from the layer after applying the activation function.
        """
        if self.weights is None:
            self.weights = np.random.randn(X.shape[1], self.node_count) / np.sqrt(X.shape[1])
            self.bias = np.zeros((1, self.node_count))

        self.last_input = X
        self.z = np.dot(X, self.weights) + self.bias
        return calc_activation_func(self.z, self.activation_func_type)
    

    def backward(self, loss_gradient):
        """
        Performs backward propagation through the layer.

        Parameters:
        loss_gradient: numpy array - The gradient of the loss with respect to the output from the layer.

        Returns:
        numpy array - The gradient of the loss with respect to the input to the layer.
        """
        dZ = None
        if self.activation_func_type == 'softmax':
            jacobian_matrix = calc_activation_func_derivative(self.z, self.activation_func_type)
            dZ = np.einsum('ij,ijk->ik', loss_gradient, jacobian_matrix)
        else:
            dZ = loss_gradient * calc_activation_func_derivative(self.z, self.activation_func_type)

        self.dW = np.dot(self.last_input.T, dZ) / self.last_input.shape[0]  # divide by batch size
        self.dB = np.sum(dZ, axis=0, keepdims=True) / self.last_input.shape[0]  # divide by batch size
        return np.dot(dZ, self.weights.T)
    

    def update(self, learning_rate):
        """
        Updates the parameters of the layer using the given learning rate.

        Parameters:
        learning_rate: float - The learning rate to use for parameter updates.
        """
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.dB
