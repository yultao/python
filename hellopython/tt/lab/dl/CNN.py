import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


class CNN():
    # GRADED FUNCTION: zero_pad
    
    def zero_pad(self, X, pad):
        """
        Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
        as illustrated in Figure 1.
        
        Argument:
        X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
        pad -- integer, amount of padding around each image on vertical and horizontal dimensions
        
        Returns:
        X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """
        
        ### START CODE HERE ### (≈ 1 line)
        X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (0,0))
        ### END CODE HERE ###
        
        return X_pad
    # GRADED FUNCTION: conv_single_step
    
    def conv_single_step(self, a_slice_prev, W, b):
        """
        Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
        of the previous layer.
        
        Arguments:
        a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
        W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
        b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
        
        Returns:
        Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
        """
    
        ### START CODE HERE ### (≈ 2 lines of code)
        # Element-wise product between a_slice and W. Do not add the bias yet.
        s = a_slice_prev * W
        # Sum over all entries of the volume s.
        Z = np.sum(s)
        # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
        Z = Z + float(b)
        ### END CODE HERE ###
    
        return Z
    
    # GRADED FUNCTION: conv_forward

    def conv_forward(self, A_prev, W, b, hparameters):
        """
        Implements the forward propagation for a convolution function
        
        Arguments:
        A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
        b -- Biases, numpy array of shape (1, 1, 1, n_C)
        hparameters -- python dictionary containing "stride" and "pad"
            
        Returns:
        Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward() function
        """
        
        ### START CODE HERE ###
        # Retrieve dimensions from A_prev's shape (≈1 line)  
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve dimensions from W's shape (≈1 line)
        (f, f, n_C_prev, n_C) = W.shape
        
        # Retrieve information from "hparameters" (≈2 lines)
        stride = hparameters["stride"]
        pad = hparameters["pad"]
        
        # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
        n_H = int((n_H_prev - f + 2 * pad)/stride) + 1
        n_W = int((n_W_prev - f + 2 * pad)/stride) + 1
        
        # Initialize the output volume Z with zeros. (≈1 line)
        Z = np.zeros((m, n_H, n_W, n_C))
        
        # Create A_prev_pad by padding A_prev
        A_prev_pad = self.zero_pad(A_prev, pad)
        
        
        for i in range(m):                                # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i,:,:,:]              # Select ith training example's padded activation
            
            for h in range(n_H):                          # loop over vertical axis of the output volume
                for w in range(n_W):                      # loop over horizontal axis of the output volume
                    for c in range(n_C):                  # loop over channels (= #filters) of the output volume
                        
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h*stride
                        vert_end = h*stride+f
                        horiz_start = w*stride
                        horiz_end = w*stride+f
                        
                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                                            
                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                        Z[i, h, w, c] = self.conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])
                        
        ### END CODE HERE ###
        
        # Making sure your output shape is correct
        assert(Z.shape == (m, n_H, n_W, n_C))
        
        # Save information in "cache" for the backprop
        cache = (A_prev, W, b, hparameters)
        
        return Z, cache
    
    # GRADED FUNCTION: pool_forward

    def pool_forward(self, A_prev, hparameters, mode = "max"):
        """
        Implements the forward pass of the pooling layer
        
        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        hparameters -- python dictionary containing "f" and "stride"
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
        
        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
        """
        
        # Retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve hyperparameters from "hparameters"
        f = hparameters["f"]
        stride = hparameters["stride"]
        
        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev
        
        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))              
        
        ### START CODE HERE ###
        for i in range(m):                         # loop over the training examples
            for h in range(n_H):                     # loop on the vertical axis of the output volume
                for w in range(n_W):                 # loop on the horizontal axis of the output volume
                    for c in range (n_C):            # loop over the channels of the output volume
                        
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h*stride
                        vert_end = h*stride+f
                        horiz_start = w*stride
                        horiz_end = w*stride+f
                        
                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                        
                        # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                        if mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
        
        ### END CODE HERE ###
        
        # Store the input and hparameters in "cache" for pool_backward()
        cache = (A_prev, hparameters)
        
        # Making sure your output shape is correct
        assert(A.shape == (m, n_H, n_W, n_C))
        
        return A, cache
    
    def conv_backward(self, dZ, cache):
        """
        Implement the backward propagation for a convolution function
        
        Arguments:
        dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward(), output of conv_forward()
        
        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                   numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
              numpy array of shape (f, f, n_C_prev, n_C)
        db -- gradient of the cost with respect to the biases of the conv layer (b)
              numpy array of shape (1, 1, 1, n_C)
        """
        
        ### START CODE HERE ###
        # Retrieve information from "cache"
        (A_prev, W, b, hparameters) = cache
        
        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = W.shape
        
        # Retrieve information from "hparameters"
        stride = hparameters["stride"]
        pad = hparameters["pad"]
        
        # Retrieve dimensions from dZ's shape
        (m, n_H, n_W, n_C) = dZ.shape
        
        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros(A_prev.shape)                           
        dW = np.zeros(W.shape) 
        db = np.zeros(b.shape) 
    
        # Pad A_prev and dA_prev
        A_prev_pad = self.zero_pad(A_prev, pad)
        dA_prev_pad = self.zero_pad(dA_prev, pad)
        
        for i in range(m):                       # loop over the training examples
            
            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i,:,:,:]
            da_prev_pad = dA_prev_pad[i,:,:,:]
            
            for h in range(n_H):                   # loop over vertical axis of the output volume
                for w in range(n_W):               # loop over horizontal axis of the output volume
                    for c in range(n_C):           # loop over the channels of the output volume
                        
                        # Find the corners of the current "slice"
                        vert_start = h*stride
                        vert_end = h*stride+f
                        horiz_start = w*stride
                        horiz_end = w*stride+f
                        
                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
    
                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                        dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                        db[:,:,:,c] += dZ[i, h, w, c]
                        
            # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        ### END CODE HERE ###
        
        # Making sure your output shape is correct
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
        
        return dA_prev, dW, db
    def create_mask_from_window(self, x):
        """
        Creates a mask from an input matrix x, to identify the max entry of x.
        
        Arguments:
        x -- Array of shape (f, f)
        
        Returns:
        mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """
        
        ### START CODE HERE ### (≈1 line)
        mask = x == np.max(x)
        ### END CODE HERE ###
        
        return mask
    def distribute_value(self, dz, shape):
        """
        Distributes the input value in the matrix of dimension shape
        
        Arguments:
        dz -- input scalar
        shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
        
        Returns:
        a -- Array of size (n_H, n_W) for which we distributed the value of dz
        """
        
        ### START CODE HERE ###
        # Retrieve dimensions from shape (≈1 line)
        (n_H, n_W) = shape
        
        # Compute the value to distribute on the matrix (≈1 line)
        average = dz / (n_H * n_W)
        
        # Create a matrix where every entry is the "average" value (≈1 line)
        a = np.ones(shape) * average
        ### END CODE HERE ###
        
        return a
    def pool_backward(self, dA, cache, mode = "max"):
        """
        Implements the backward pass of the pooling layer
        
        Arguments:
        dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
        cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
        
        Returns:
        dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
        """
        
        ### START CODE HERE ###
        
        # Retrieve information from cache (≈1 line)
        (A_prev, hparameters) = cache
        
        # Retrieve hyperparameters from "hparameters" (≈2 lines)
        stride = hparameters["stride"]
        f = hparameters["f"]
        
        # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape
        
        # Initialize dA_prev with zeros (≈1 line)
        dA_prev = np.zeros(A_prev.shape)
        
        for i in range(m):                       # loop over the training examples
            
            # select training example from A_prev (≈1 line)
            a_prev = A_prev[i,:,:,:]
            
            for h in range(n_H):                   # loop on the vertical axis
                for w in range(n_W):               # loop on the horizontal axis
                    for c in range(n_C):           # loop over the channels (depth)
                        
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h*stride
                        vert_end = h*stride+f
                        horiz_start = w*stride
                        horiz_end = w*stride+f
                        
                        # Compute the backward propagation in both modes.
                        if mode == "max":
                            
                            # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                            a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                            # Create the mask from a_prev_slice (≈1 line)
                            mask = self.create_mask_from_window(a_prev_slice)
                            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask*dA[i,h:h+1,w:w+1,c]
                            
                        elif mode == "average":
                            
                            # Get the value a from dA (≈1 line)
                            da = dA[i,h:h+1,w:w+1,c]
                            # Define the shape of the filter as fxf (≈1 line)
                            shape = (f,f)
                            # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += self.distribute_value(da, shape)
                            
        ### END CODE ###
        
        # Making sure your output shape is correct
        assert(dA_prev.shape == A_prev.shape)
        
        return dA_prev
## SimpleConvolutionalNeuralNetwork ##################
nn = CNN()
class CNNTestCase():
#     x.shape = (4, 3, 3, 2)
#     x_pad.shape = (4, 7, 7, 2)
#     x[1,1] = [[ 0.90085595 -0.68372786]
#      [-0.12289023 -0.93576943]
#      [-0.26788808  0.53035547]]
#     x_pad[1,1] = [[ 0.  0.]
#      [ 0.  0.]
#      [ 0.  0.]
#      [ 0.  0.]
#      [ 0.  0.]
#      [ 0.  0.]
#      [ 0.  0.]]
    def test_zero_pad(self):
        np.random.seed(1)
        x = np.random.randn(4, 3, 3, 2)
        x_pad = nn.zero_pad(x, 2)
        print ("x.shape =", x.shape)
        print ("x_pad.shape =", x_pad.shape)
        print ("x[1,1] =", x[1,1])
        print ("x_pad[1,1] =", x_pad[1,1])
        
        fig, axarr = plt.subplots(1, 2)
        axarr[0].set_title('x')
        axarr[0].imshow(x[0,:,:,0])
        axarr[1].set_title('x_pad')
        axarr[1].imshow(x_pad[0,:,:,0])
        plt.show()
#     Z = -6.99908945068
    def test_conv_single_step(self):
        np.random.seed(1)
        a_slice_prev = np.random.randn(4, 4, 3)
        W = np.random.randn(4, 4, 3)
        b = np.random.randn(1, 1, 1)
        
        Z = nn.conv_single_step(a_slice_prev, W, b)
        print("Z =", Z)
        
#     Z's mean = 0.0489952035289
#     Z[3,2,1] = [-0.61490741 -6.7439236  -2.55153897  1.75698377  3.56208902  0.53036437
#       5.18531798  8.75898442]
#     cache_conv[0][1][2][3] = [-0.20075807  0.18656139  0.41005165]
    def test_conv_forward(self):
        np.random.seed(1)
        A_prev = np.random.randn(10,4,4,3)
        W = np.random.randn(2,2,3,8)
        b = np.random.randn(1,1,1,8)
        hparameters = {"pad" : 2,
                       "stride": 2}
        
        Z, cache_conv = nn.conv_forward(A_prev, W, b, hparameters)
        print("Z's mean =", np.mean(Z))
        print("Z[3,2,1] =", Z[3,2,1])
        print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
        return (Z, cache_conv)
#     mode = max
#     A = [[[[ 1.74481176  0.86540763  1.13376944]]]
#     
#     
#      [[[ 1.13162939  1.51981682  2.18557541]]]]
#     
#     mode = average
#     A = [[[[ 0.02105773 -0.20328806 -0.40389855]]]
#     
#     
#      [[[-0.22154621  0.51716526  0.48155844]]]]

    def test_pool_forward(self):
        np.random.seed(1)
        A_prev = np.random.randn(2, 4, 4, 3)
        hparameters = {"stride" : 2, "f": 3}
        
        A, cache = nn.pool_forward(A_prev, hparameters)
        print("mode = max")
        print("A =", A)
        print()
        A, cache = nn.pool_forward(A_prev, hparameters, mode = "average")
        print("mode = average")
        print("A =", A)
        
#     dA_mean = 1.45243777754
#     dW_mean = 1.72699145831
#     db_mean = 7.83923256462

    def test_conv_backward(self):
        Z, cache_conv = self.test_conv_forward()
        np.random.seed(1)
        dA, dW, db = nn.conv_backward(Z, cache_conv)
        print("dA_mean =", np.mean(dA))
        print("dW_mean =", np.mean(dW))
        print("db_mean =", np.mean(db))
#         x =  [[ 1.62434536 -0.61175641 -0.52817175]
#          [-1.07296862  0.86540763 -2.3015387 ]]
#         mask =  [[ True False False]
#          [False False False]]
    def test_create_mask_from_window(self):
        np.random.seed(1)
        x = np.random.randn(2,3)
        mask = nn.create_mask_from_window(x)
        print('x = ', x)
        print("mask = ", mask)
#     distributed value = [[ 0.5  0.5]
#      [ 0.5  0.5]]

    def test_distribute_value(self):
        a = nn.distribute_value(2, (2,2))
        print('distributed value =', a)
        
#     mode = max
#     mean of dA =  0.145713902729
#     dA_prev[1,1] =  [[ 0.          0.        ]
#      [ 5.05844394 -1.68282702]
#      [ 0.          0.        ]]
#     
#     mode = average
#     mean of dA =  0.145713902729
#     dA_prev[1,1] =  [[ 0.08485462  0.2787552 ]
#      [ 1.26461098 -0.25749373]
#      [ 1.17975636 -0.53624893]]

    def test_pool_backward(self):
        np.random.seed(1)
        A_prev = np.random.randn(5, 5, 3, 2)
        hparameters = {"stride" : 1, "f": 2}
        A, cache = nn.pool_forward(A_prev, hparameters)
        dA = np.random.randn(5, 4, 2, 2)
        
        dA_prev = nn.pool_backward(dA, cache, mode = "max")
        print("mode = max")
        print('mean of dA = ', np.mean(dA))
        print('dA_prev[1,1] = ', dA_prev[1,1])  
        print()
        dA_prev = nn.pool_backward(dA, cache, mode = "average")
        print("mode = average")
        print('mean of dA = ', np.mean(dA))
        print('dA_prev[1,1] = ', dA_prev[1,1]) 
## SimpleConvolutionalNeuralNetworkTestCase ##################
if __name__ == '__main__':
    tc = CNNTestCase();
#     tc.test_zero_pad();
#     tc.test_conv_single_step();
#     tc.test_conv_forward();
#     tc.test_pool_forward();
#     tc.test_conv_backward();
#     tc.test_create_mask_from_window();
#     tc.test_distribute_value();
    tc.test_pool_backward();