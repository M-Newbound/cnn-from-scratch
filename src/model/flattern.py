
class Flatten:
    """
    Represents a layer that flattens the output of a convolutional layer in a Neural Network.
    """

    def __init__(self):
        """
        Initializes the Flatten layer.
        """
        pass

    def forward(self, input):
        """
        Performs forward propagation through the layer by flattening the input.

        Parameters:
        input: numpy array of shape (batch_size, h, w, num_filters) - The input data.

        Returns:
        numpy array of shape (batch_size, h * w * num_filters) - The flattened output.
        """
        self.last_input_shape = input.shape

        batch_size, h, w, num_filters = input.shape
        output = input.reshape(batch_size, h * w * num_filters)

        return output


    def backward(self, d_L_d_out):
        """
        Performs backward propagation through the layer by reshaping the input back to its original shape.

        Parameters:
        d_L_d_out: numpy array - The gradient of the loss with respect to the output from the layer.

        Returns:
        numpy array of the original shape - The reshaped input.
        """
        return d_L_d_out.T.reshape(self.last_input_shape)
    

    def update(self, learning_rate):
        """
        No update operation needed for this layer as it doesn't have any learnable parameters.
        """
        pass