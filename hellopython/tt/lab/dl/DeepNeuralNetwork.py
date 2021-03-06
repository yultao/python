import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.misc as misc
from scipy import ndimage

"""
Test cases
"""
class DeepNeuralNetworkTestCase():
    def linear_forward_test_case(self):
        np.random.seed(1)
        """
        X = np.array([[-1.02387576, 1.12397796],
     [-1.62328545, 0.64667545],
     [-1.74314104, -0.59664964]])
        W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
        b = np.array([[1]])
        """
        A = np.random.randn(3,2)
        W = np.random.randn(1,3)
        b = np.random.randn(1,1)
        
        return A, W, b
    
    def linear_activation_forward_test_case(self):
        """
        X = np.array([[-1.02387576, 1.12397796],
     [-1.62328545, 0.64667545],
     [-1.74314104, -0.59664964]])
        W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
        b = 5
        """
        np.random.seed(2)
        A_prev = np.random.randn(3,2)
        W = np.random.randn(1,3)
        b = np.random.randn(1,1)
        return A_prev, W, b
    
    def L_model_forward_test_case(self):
        """
        X = np.array([[-1.02387576, 1.12397796],
     [-1.62328545, 0.64667545],
     [-1.74314104, -0.59664964]])
        parameters = {'W1': np.array([[ 1.62434536, -0.61175641, -0.52817175],
            [-1.07296862,  0.86540763, -2.3015387 ]]),
     'W2': np.array([[ 1.74481176, -0.7612069 ]]),
     'b1': np.array([[ 0.],
            [ 0.]]),
     'b2': np.array([[ 0.]])}
        """
        np.random.seed(1)
        X = np.random.randn(4,2)
        W1 = np.random.randn(3,4)
        b1 = np.random.randn(3,1)
        W2 = np.random.randn(1,3)
        b2 = np.random.randn(1,1)
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return X, parameters
    
    def compute_cost_test_case(self):
        Y = np.asarray([[1, 1, 1]])
        aL = np.array([[.8,.9,0.4]])
        
        return Y, aL
    
    def linear_backward_test_case(self):
        """
        z, linear_cache = (np.array([[-0.8019545 ,  3.85763489]]), (np.array([[-1.02387576,  1.12397796],
           [-1.62328545,  0.64667545],
           [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), np.array([[1]]))
        """
        np.random.seed(1)
        dZ = np.random.randn(1,2)
        A = np.random.randn(3,2)
        W = np.random.randn(1,3)
        b = np.random.randn(1,1)
        linear_cache = (A, W, b)
        return dZ, linear_cache
    
    def linear_activation_backward_test_case(self):
        """
        aL, linear_activation_cache = (np.array([[ 3.1980455 ,  7.85763489]]), ((np.array([[-1.02387576,  1.12397796], [-1.62328545,  0.64667545], [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), 5), np.array([[ 3.1980455 ,  7.85763489]])))
        """
        np.random.seed(2)
        dA = np.random.randn(1,2)
        A = np.random.randn(3,2)
        W = np.random.randn(1,3)
        b = np.random.randn(1,1)
        Z = np.random.randn(1,2)
        linear_cache = (A, W, b)
        activation_cache = Z
        linear_activation_cache = (linear_cache, activation_cache)
        
        return dA, linear_activation_cache
    
    def L_model_backward_test_case(self):
        """
        X = np.random.rand(3,2)
        Y = np.array([[1, 1]])
        parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747]]), 'b1': np.array([[ 0.]])}
    
        aL, caches = (np.array([[ 0.60298372,  0.87182628]]), [((np.array([[ 0.20445225,  0.87811744],
               [ 0.02738759,  0.67046751],
               [ 0.4173048 ,  0.55868983]]),
        np.array([[ 1.78862847,  0.43650985,  0.09649747]]),
        np.array([[ 0.]])),
       np.array([[ 0.41791293,  1.91720367]]))])
       """
        np.random.seed(3)
        AL = np.random.randn(1, 2)
        Y = np.array([[1, 0]])
    
        A1 = np.random.randn(4,2)
        W1 = np.random.randn(3,4)
        b1 = np.random.randn(3,1)
        Z1 = np.random.randn(3,2)
        linear_cache_activation_1 = ((A1, W1, b1), Z1)
    
        A2 = np.random.randn(3,2)
        W2 = np.random.randn(1,3)
        b2 = np.random.randn(1,1)
        Z2 = np.random.randn(1,2)
        linear_cache_activation_2 = ((A2, W2, b2), Z2)
    
        caches = (linear_cache_activation_1, linear_cache_activation_2)
    
        return AL, Y, caches
    
    def update_parameters_test_case(self):
        """
        parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747],
            [-1.8634927 , -0.2773882 , -0.35475898],
            [-0.08274148, -0.62700068, -0.04381817],
            [-0.47721803, -1.31386475,  0.88462238]]),
     'W2': np.array([[ 0.88131804,  1.70957306,  0.05003364, -0.40467741],
            [-0.54535995, -1.54647732,  0.98236743, -1.10106763],
            [-1.18504653, -0.2056499 ,  1.48614836,  0.23671627]]),
     'W3': np.array([[-1.02378514, -0.7129932 ,  0.62524497],
            [-0.16051336, -0.76883635, -0.23003072]]),
     'b1': np.array([[ 0.],
            [ 0.],
            [ 0.],
            [ 0.]]),
     'b2': np.array([[ 0.],
            [ 0.],
            [ 0.]]),
     'b3': np.array([[ 0.],
            [ 0.]])}
        grads = {'dW1': np.array([[ 0.63070583,  0.66482653,  0.18308507],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]]),
     'dW2': np.array([[ 1.62934255,  0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ,  0.        ]]),
     'dW3': np.array([[-1.40260776,  0.        ,  0.        ]]),
     'da1': np.array([[ 0.70760786,  0.65063504],
            [ 0.17268975,  0.15878569],
            [ 0.03817582,  0.03510211]]),
     'da2': np.array([[ 0.39561478,  0.36376198],
            [ 0.7674101 ,  0.70562233],
            [ 0.0224596 ,  0.02065127],
            [-0.18165561, -0.16702967]]),
     'da3': np.array([[ 0.44888991,  0.41274769],
            [ 0.31261975,  0.28744927],
            [-0.27414557, -0.25207283]]),
     'db1': 0.75937676204411464,
     'db2': 0.86163759922811056,
     'db3': -0.84161956022334572}
        """
        np.random.seed(2)
        W1 = np.random.randn(3,4)
        b1 = np.random.randn(3,1)
        W2 = np.random.randn(1,3)
        b2 = np.random.randn(1,1)
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        np.random.seed(3)
        dW1 = np.random.randn(3,4)
        db1 = np.random.randn(3,1)
        dW2 = np.random.randn(1,3)
        db2 = np.random.randn(1,1)
        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
        
        return parameters, grads
    
    
    def L_model_forward_test_case_2hidden(self):
        np.random.seed(6)
        X = np.random.randn(5,4)
        W1 = np.random.randn(4,5)
        b1 = np.random.randn(4,1)
        W2 = np.random.randn(3,4)
        b2 = np.random.randn(3,1)
        W3 = np.random.randn(1,3)
        b3 = np.random.randn(1,1)
      
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}
        
        return X, parameters
    
    def print_grads(self, grads):
        print ("dW1 = "+ str(grads["dW1"]))
        print ("db1 = "+ str(grads["db1"]))
        print ("dA1 = "+ str(grads["dA2"])) # this is done on purpose to be consistent with lecture where we normally start with A0
                                        # in this implementation we started with A1, hence we bump it up by 1. 
    
class DeepNeuralNetwork():
    def load_data(self):
        train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    
        test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    
        classes = np.array(test_dataset["list_classes"][:]) # the list of classes
        
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes    
    def sigmoid(self, Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
        
        return A, cache
    
    def relu(self, Z):
        """
        Implement the RELU function.
    
        Arguments:
        Z -- Output of the linear layer, of any shape
    
        Returns:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
        """
        
        A = np.maximum(0,Z)
        
        assert(A.shape == Z.shape)
        
        cache = Z 
        return A, cache
    
    
    def relu_backward(self, dA, cache):
        """
        Implement the backward propagation for a single RELU unit.
    
        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently
    
        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        
        assert (dZ.shape == Z.shape)
        
        return dZ
    
    def sigmoid_backward(self, dA, cache):
        """
        Implement the backward propagation for a single SIGMOID unit.
    
        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently
    
        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        Z = cache
        
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
        assert (dZ.shape == Z.shape)
        
        return dZ
    # GRADED FUNCTION: initialize_parameters
    
    def initialize_parameters(self, n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        
        Returns:
        parameters -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """
        
        np.random.seed(1)
        
        ### START CODE HERE ### (�� 4 lines of code)
        W1 = np.random.randn(n_h,n_x)*0.01
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h)*0.01
        b2 = np.zeros((n_y,1))
        ### END CODE HERE ###
        
        assert(W1.shape == (n_h, n_x))
        assert(b1.shape == (n_h, 1))
        assert(W2.shape == (n_y, n_h))
        assert(b2.shape == (n_y, 1))
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return parameters    
    
    # GRADED FUNCTION: initialize_parameters_deep

    def initialize_parameters_deep(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        
        np.random.seed(1) # GT: change from 3 to 1
        parameters = {}
        L = len(layer_dims)            # number of layers in the network
    
        for l in range(1, L):
            ### START CODE HERE ### (≈ 2 lines of code)
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01 GT: remove 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            ### END CODE HERE ###
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
            
        return parameters
    # GRADED FUNCTION: linear_forward
    
    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.
    
        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
    
        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
        
        ### START CODE HERE ### (≈ 1 line of code)
        Z = np.dot(W,A)+b
        ### END CODE HERE ###
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache
    # GRADED FUNCTION: linear_activation_forward
    
    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer
    
        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """
        
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
            ### END CODE HERE ###
        
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
            ### END CODE HERE ###
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
    
        return A, cache
    
    # GRADED FUNCTION: L_model_forward
    
    def L_model_forward(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """
    
        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            ### START CODE HERE ### (≈ 2 lines of code)
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
            caches.append(cache)
            ### END CODE HERE ###
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        ### START CODE HERE ### (≈ 2 lines of code)
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
        caches.append(cache)
        ### END CODE HERE ###
        
        assert(AL.shape == (1,X.shape[1]))
                
        return AL, caches
    # GRADED FUNCTION: compute_cost
    
    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).
    
        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    
        Returns:
        cost -- cross-entropy cost
        """
        
        m = Y.shape[1]
    
        # Compute loss from aL and y.
        ### START CODE HERE ### (≈ 1 lines of code)
        cost = -(1/m) * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
        ### END CODE HERE ###
        
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        
        return cost    
    # GRADED FUNCTION: linear_backward
    
    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)
    
        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]
    
        ### START CODE HERE ### (≈ 3 lines of code)
        dW = (1/m)*np.dot(dZ, A_prev.T)
        db = (1/m)*np.sum(dZ, axis=1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)
        ### END CODE HERE ###
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db    
    # GRADED FUNCTION: linear_activation_backward
    
    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###
            
        elif activation == "sigmoid":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###
        
        return dA_prev, dW, db    
    
    # GRADED FUNCTION: L_model_backward
    
    def L_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        
        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        ### START CODE HERE ### (1 line of code)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        ### END CODE HERE ###
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        ### START CODE HERE ### (approx. 2 lines)
        current_cache = caches[L-1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid")
        ### END CODE HERE ###
        
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            ### START CODE HERE ### (approx. 5 lines)
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            ### END CODE HERE ###
    
        return grads  
    
    # GRADED FUNCTION: update_parameters

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                      parameters["W" + str(l)] = ... 
                      parameters["b" + str(l)] = ...
        """
        
        L = len(parameters) // 2 # number of layers in the neural network
    
        # Update rule for each parameter. Use a for loop.
        ### START CODE HERE ### (≈ 3 lines of code)
        for l in range(1, L+1):
            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        ### END CODE HERE ###
        return parameters
    
    
    # GRADED FUNCTION: two_layer_model

    def two_layer_model(self, X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
        """
        Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
        
        Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- dimensions of the layers (n_x, n_h, n_y)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- If set to True, this will print the cost every 100 iterations 
        
        Returns:
        parameters -- a dictionary containing W1, W2, b1, and b2
        """
        
        np.random.seed(1)
        grads = {}
        costs = []                              # to keep track of the cost
        m = X.shape[1]                           # number of examples
        (n_x, n_h, n_y) = layers_dims
        
        # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = self.initialize_parameters(n_x, n_h, n_y)
        ### END CODE HERE ###
        
        # Get W1, b1, W2 and b2 from the dictionary parameters.
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Loop (gradient descent)
    
        for i in range(0, num_iterations):
    
            # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
            ### START CODE HERE ### (≈ 2 lines of code)
            A1, cache1 = self.linear_activation_forward(X, W1, b1, "relu")
            A2, cache2 = self.linear_activation_forward(A1, W2, b2, "sigmoid")
            ### END CODE HERE ###
            
            # Compute cost
            ### START CODE HERE ### (≈ 1 line of code)
            cost = self.compute_cost(A2, Y)
            ### END CODE HERE ###
            
            # Initializing backward propagation
            dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
            
            # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
            ### START CODE HERE ### (≈ 2 lines of code)
            dA1, dW2, db2 = self.linear_activation_backward(dA2, cache2, "sigmoid")
            dA0, dW1, db1 = self.linear_activation_backward(dA1, cache1, "relu")
            ### END CODE HERE ###
            
            # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
            grads['dW1'] = dW1
            grads['db1'] = db1
            grads['dW2'] = dW2
            grads['db2'] = db2
            
            # Update parameters.
            ### START CODE HERE ### (approx. 1 line of code)
            parameters = self.update_parameters(parameters, grads, learning_rate)
            ### END CODE HERE ###
    
            # Retrieve W1, b1, W2, b2 from parameters
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
            
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if print_cost and i % 100 == 0:
                costs.append(cost)
           
        # plot the cost
    
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters
    
    def predict(self, X, y, parameters):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.L_model_forward(X, parameters)
    
        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        print("Accuracy: "  + str(np.sum((p == y)/m)))
            
        return p
    
    # GRADED FUNCTION: L_layer_model
    def L_layer_model(self, X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        
        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
    
        np.random.seed(1)
        costs = []                         # keep track of cost
        
        # Parameters initialization.
        ### START CODE HERE ###
        parameters = self.initialize_parameters_deep(layers_dims)
        ### END CODE HERE ###
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):
    
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            ### START CODE HERE ### (≈ 1 line of code)
            AL, caches = self.L_model_forward(X, parameters)
            ### END CODE HERE ###
            
            # Compute cost.
            ### START CODE HERE ### (≈ 1 line of code)
            cost = self.compute_cost(AL, Y)
            ### END CODE HERE ###
        
            # Backward propagation.
            ### START CODE HERE ### (≈ 1 line of code)
            grads = self.L_model_backward(AL, Y, caches)
            ### END CODE HERE ###
     
            # Update parameters.
            ### START CODE HERE ### (≈ 1 line of code)
            parameters = self.update_parameters(parameters, grads, learning_rate)
            ### END CODE HERE ###
                    
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters
        
    def print_mislabeled_images(self, classes, X, y, p):
        """
        Plots images where predictions and truth were different.
        X -- dataset
        y -- true labels
        p -- predictions
        """
        a = p + y
        mislabeled_indices = np.asarray(np.where(a == 1))
        plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
        num_images = len(mislabeled_indices[0])
        for i in range(num_images):
            index = mislabeled_indices[1][i]
            
            plt.subplot(2, num_images, i + 1)
            plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
            plt.axis('off')
            plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
        plt.show()    
######################3
"""
Test
"""

if __name__ == '__main__':
    np.random.seed(1)
    plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    dnn = DeepNeuralNetwork()
    dnntc = DeepNeuralNetworkTestCase()
    print("\n===================initialize_parameters\n")
    parameters = dnn.initialize_parameters(3,2,1)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    
    print("\n===================initialize_parameters_deep\n")
    parameters = dnn.initialize_parameters_deep([5,4,3])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    print("\n===================linear_forward\n")
    A, W, b = dnntc.linear_forward_test_case()

    Z, linear_cache = dnn.linear_forward(A, W, b)
    print("Z = " + str(Z))
    
    print("\n===================linear_activation_forward\n")
    A_prev, W, b = dnntc.linear_activation_forward_test_case()

    A, linear_activation_cache = dnn.linear_activation_forward(A_prev, W, b, activation = "sigmoid")
    print("With sigmoid: A = " + str(A))
    
    A, linear_activation_cache = dnn.linear_activation_forward(A_prev, W, b, activation = "relu")
    print("With ReLU: A = " + str(A))
    
    print("\n===================L_model_forward\n")
    X, parameters = dnntc.L_model_forward_test_case_2hidden()
    AL, caches = dnn.L_model_forward(X, parameters)
    print("AL = " + str(AL))
    print("Length of caches list = " + str(len(caches)))
    
    print("\n===================compute_cost\n")
    Y, AL = dnntc.compute_cost_test_case()

    print("cost = " + str(dnn.compute_cost(AL, Y)))
    
    print("\n===================linear_backward\n")
    # Set up some test inputs
    dZ, linear_cache = dnntc.linear_backward_test_case()
    
    dA_prev, dW, db = dnn.linear_backward(dZ, linear_cache)
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))
    
    print("\n===================linear_activation_backward\n")
    AL, linear_activation_cache = dnntc.linear_activation_backward_test_case()

    dA_prev, dW, db = dnn.linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
    print ("sigmoid:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db) + "\n")
    
    dA_prev, dW, db = dnn.linear_activation_backward(AL, linear_activation_cache, activation = "relu")
    print ("relu:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))
    
    print("\n===================L_model_backward\n")
    AL, Y_assess, caches = dnntc.L_model_backward_test_case()
    grads = dnn.L_model_backward(AL, Y_assess, caches)
    dnntc.print_grads(grads)
    
    print("\n===================update_parameters\n")
    parameters, grads = dnntc.update_parameters_test_case()
    parameters = dnn.update_parameters(parameters, grads, 0.1)
    
    print ("W1 = "+ str(parameters["W1"]))
    print ("b1 = "+ str(parameters["b1"]))
    print ("W2 = "+ str(parameters["W2"]))
    print ("b2 = "+ str(parameters["b2"]))
    
    print("\n===================load_data\n")
    train_x_orig, train_y, test_x_orig, test_y, classes = dnn.load_data()
    
    # Example of a picture
    index = 10
    plt.imshow(train_x_orig[index])
    print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
    plt.show()
    
    print("\n===================Explore your dataset \n")
    # Explore your dataset 
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]
    
    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_y shape: " + str(test_y.shape))
    
    print("\n===================Reshape\n")
    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.
    
    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))
    
    
    ### CONSTANTS DEFINING THE MODEL ####
    n_x = 12288     # num_px * num_px * 3
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)
    
#     print("\n===================two_layer_model\n")
#     parameters = dnn.two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
#     
#     print("\n===================predict train\n")
#     pred_train = dnn.predict(train_x, train_y, parameters)
#     
#     print("\n===================predict test\n")
#     pred_test = dnn.predict(test_x, test_y, parameters)
    
    ### CONSTANTS ###
    layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
    print("\n===================L_layer_model\n")
    parameters = dnn.L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
    
    print("\n===================predict train\n")
    pred_train = dnn.predict(train_x, train_y, parameters)
    
    print("\n===================predict test\n")
    pred_test = dnn.predict(test_x, test_y, parameters)
    
    
    
    print("\n===================print_mislabeled_images\n")
    dnn.print_mislabeled_images(classes, test_x, test_y, pred_test)
    
    print("\n===================predict my own img\n")
    ## START CODE HERE ##
    my_image = "deepcat.jpg" # change this to the name of your image file 
    my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
    ## END CODE HERE ##
    
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
    my_predicted_image = dnn.predict(my_image, my_label_y, parameters)
    
    plt.imshow(image)
    print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    plt.show()