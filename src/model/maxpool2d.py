
import numpy as np

class MaxPool2D:
    """
    Represents a 2D max pooling layer in a Convolutional Neural Network.
    """

    def __init__(self, pool_size):
        """
        Initializes the MaxPool2D layer.

        Parameters:
        pool_size: int - The size of the pooling window.
        """
        self.pool_size = pool_size

    def iterate_regions(self, image):
        """
        Generates all possible square image regions using the pool size.

        Parameters:
        image: numpy array - The input image.

        Yields:
        tuple - The image region and its position.
        """
        batch_size, h, w, _ = image.shape
        new_h = h // self.pool_size
        new_w = w // self.pool_size

        for b in range(batch_size):
            for i in range(new_h):
                for j in range(new_w):
                    im_region = image[b, (i * self.pool_size):(i * self.pool_size + self.pool_size), 
                                      (j * self.pool_size):(j * self.pool_size + self.pool_size)]
                    yield im_region, b, i, j

    def forward(self, input):
        """
        Performs forward propagation through the layer by applying max pooling.

        Parameters:
        input: numpy array - The input data.

        Returns:
        numpy array - The output from the layer after max pooling.
        """
        self.last_input = input

        batch_size, h, w, num_filters = input.shape
        output = np.zeros((batch_size, h // self.pool_size, w // self.pool_size, num_filters))

        for im_region, b, i, j in self.iterate_regions(input):
            output[b, i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backward(self, d_L_d_out):
        """
        Performs backward propagation through the layer by distributing the gradient where the maximum value was found.

        Parameters:
        d_L_d_out: numpy array - The gradient of the loss with respect to the output from the layer.

        Returns:
        numpy array - The gradient of the loss with respect to the input to the layer.
        """
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, b, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it.
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[b, i * self.pool_size + i2, j * self.pool_size + j2, f2] = d_L_d_out[b, i, j, f2]

        return d_L_d_input
    
    def update(self, learning_rate):
        """
        No update operation needed for this layer as it doesn't have any learnable parameters.
        """
        pass