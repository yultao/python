import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.misc as misc
from scipy import ndimage
from sklearn.datasets import fetch_mldata
import math
"""
Test cases
"""
class NNMnistTestCase():
    def load_cat_data(self):
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
    def load_mnist_data(self):
        mnist = fetch_mldata('MNIST original')
#         print(mnist.data.shape)
#         print(mnist.target.shape)
        return mnist;
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
    def random_mini_batches_test_case(self):
        np.random.seed(1)
        mini_batch_size = 64
        X = np.random.randn(12288, 148)
        Y = np.random.randn(1, 148) < 0.5
        return X, Y, mini_batch_size

class NNMnist():
    # GRADED FUNCTION: random_mini_batches
    
    def random_mini_batches(self, X, Y, mini_batch_size = 64, seed = 0):
        np.random.seed(seed)            # To make your "random" minibatches the same as ours
        m = X.shape[1]                  # number of training examples
        mini_batches = []
            
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
    
        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[:, mini_batch_size*k : (k+1)*mini_batch_size]
            mini_batch_Y = shuffled_Y[:, mini_batch_size*k : (k+1)*mini_batch_size]
            ### END CODE HERE ###
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size : m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size : m]
            ### END CODE HERE ###
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches
    def sigmoid(self, Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
        return A, cache
    def relu(self, Z):
        
        A = np.maximum(0,Z)
        
        assert(A.shape == Z.shape)
        
        cache = Z 
        return A, cache
    
    def relu_backward(self, dA, cache):
        
        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        
        assert (dZ.shape == Z.shape)
        
        return dZ
    
    def sigmoid_backward(self, dA, cache):
        
        Z = cache
        
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
        assert (dZ.shape == Z.shape)
        
        return dZ
    
    # GRADED FUNCTION: initialize_parameters_deep

    def initialize_parameters_deep(self, layer_dims):

        
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)            # number of layers in the network
    
        for l in range(1, L):
            ### START CODE HERE ### (≈ 2 lines of code)
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) *0.01 #GT: remove 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            ### END CODE HERE ###
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
            
        return parameters
    
    # GRADED FUNCTION: initialize_parameters_deep

    def initialize_parameters_deep_L_layer_model(self, layer_dims):
      
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
    
    # GRADED FUNCTION: initialize_parameters_he
    
    def initialize_parameters_he(self, layers_dims):
       
        np.random.seed(1)
        parameters = {}
        L = len(layers_dims) - 1 # integer representing the number of layers
         
        for l in range(1, L + 1):
            ### START CODE HERE ### (≈ 2 lines of code)
            parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
            ### END CODE HERE ###
            
        return parameters
    # GRADED FUNCTION: linear_forward
    
    def linear_forward(self, A_prev, W, b): #GT: change A to A_prev to avoid confusion
        
        
        ### START CODE HERE ### (≈ 1 line of code)
        Z = np.dot(W,A_prev)+b
        ### END CODE HERE ###
        
        assert(Z.shape == (W.shape[0], A_prev.shape[1]))
        cache = (A_prev, W, b)
        
        return Z, cache
    # GRADED FUNCTION: linear_activation_forward
    
    def linear_activation_forward(self, A_prev, W, b, activation):
        
        
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
        cache = (linear_cache, activation_cache)#(A_prev, W, b, Z)
    
        return A, cache
    
    # GRADED FUNCTION: L_model_forward
    
    def L_model_forward(self, X, parameters):
        
    
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
        
        
        if(len(np.where(AL == 0)[0])!=0):
#             print("Has 0 compute_cost")
            zero = np.where(AL == 0)
            AL[zero]=0.00000000001
        if(len(np.where(AL == 1)[0])!=0):
#             print("Has 1 compute_cost")
            one = np.where(AL == 1)
#             print(AL[np.where(AL == 1)[0],np.where(AL == 1)[1]])
            AL[one]=1-0.00000000001
#             print(AL[np.where(AL == 1)[0],np.where(AL == 1)[1]])
        
        
#         assert(AL.shape == (1,X.shape[1]))
                
        return AL, caches
    
    def compute_cost(self, AL, Y, parameters, lambd):
        m = Y.shape[1]
        
        # Compute loss from aL and y.
        ### START CODE HERE ### (≈ 1 lines of code)
#         AL += 0.000000001



#         print(AL)
#         print(parameters["W1"][:,0:10])
        cost = -(1/m) * np.sum(     Y * np.nan_to_num(np.log(AL)) +   (1-Y) * np.nan_to_num(np.log(1-AL))  ) 
        
        cost= np.nan_to_num(cost)
        #calc cost reg
        L2_regularization_cost = 0
        L = len(parameters)//2
        for l in range(1, L + 1):
            W = parameters['W' + str(l)]
            L2_regularization_cost += np.sum(np.square(W))
#         W1 = parameters["W1"]
#         W2 = parameters["W2"]
#         W3 = parameters["W3"]
#         L2_regularization_cost = (1/m)*(lambd/2)*(np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
        L2_regularization_cost = (1/m)*(lambd/2) * L2_regularization_cost
        
        
        cost = cost + L2_regularization_cost
        ### END CODE HERE ###
        
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        
        return cost 
    
    def linear_backward(self, dZ, cache, lambd):
        
        A_prev, W, b = cache
        m = A_prev.shape[1]
    
        ### START CODE HERE ### (≈ 3 lines of code)
        dW = (1/m)*np.dot(dZ, A_prev.T) + (lambd/m)*W
        db = (1/m)*np.sum(dZ, axis=1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)
        ### END CODE HERE ###
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db    
    
    # GRADED FUNCTION: linear_activation_backward
    
    def linear_activation_backward(self, dA, cache, activation, lambd):
        
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache, lambd)
            ### END CODE HERE ###
            
        elif activation == "sigmoid":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache, lambd)
            ### END CODE HERE ###
        
        return dA_prev, dW, db    
    
    # GRADED FUNCTION: L_model_backward
    
    def L_model_backward(self, AL, Y, caches, lambd):
        
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        ### START CODE HERE ### (1 line of code)
        if(len(np.where(AL == 0)[0])!=0):
            print("Has 0 L_model_backward")
        if(len(np.where(AL == 1)[0])!=0):
            print("Has 1 L_model_backward")
        dAL = - (np.nan_to_num(np.divide(Y, AL)) - np.nan_to_num(np.divide(1 - Y, 1 - AL)))
        ### END CODE HERE ###
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        ### START CODE HERE ### (approx. 2 lines)
        current_cache = caches[L-1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid", lambd)
        ### END CODE HERE ###
        
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            ### START CODE HERE ### (approx. 5 lines)
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu", lambd)
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            ### END CODE HERE ###
    
        return grads  

    def update_parameters(self, parameters, grads, learning_rate):
        
        L = len(parameters) // 2 # number of layers in the neural network
    
        # Update rule for each parameter. Use a for loop.
        ### START CODE HERE ### (≈ 3 lines of code)
        for l in range(1, L+1):
            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        ### END CODE HERE ###
        return parameters
    
    
    
    def predict(self, X, y, parameters):

        
        m = X.shape[1]
        n = len(parameters) // 2 # number of layers in the neural network
        p = np.zeros((y.shape[0],m))
        
        # Forward propagation
        probas, caches = self.L_model_forward(X, parameters)
    
        
        # convert probas and y to 0-9 predictions
        p = np.argmax(probas, axis=0).reshape(1,m)
        y_orig = np.argmax(y, axis=0).reshape(1,m)

        print("Accuracy: "  + str(np.sum((p == y_orig)/m)))
            
        return p
    
    
    # GRADED FUNCTION: L_layer_model
    def L_layer_model(self, X, Y, layers_dims, learning_rate = 0.0075, 
                                          print_cost=False, lambd=0,
                                          num_epochs = 1000, mini_batch_size = 64,
                                          print_cost_interval=1
                                          ):
        
        np.random.seed(1)
        costs = []                         # keep track of cost
        
        # Parameters initialization.
        ### START CODE HERE ###
        parameters = self.initialize_parameters_deep_L_layer_model(layers_dims)
        ### END CODE HERE ###
        seed = 10
        # Loop (gradient descent)
        for i in range(num_epochs):
            print("epcho: "+str(i))
            seed = seed + 1
            minibatches = self.random_mini_batches(X, Y, mini_batch_size, seed)
            num_minibatches = len(minibatches)
            num_mb = 0
            
            for minibatch in minibatches:
#                 num_mb += 1
#                 print(str(num_mb)+"/"+str(num_minibatches))
                
                (minibatch_X, minibatch_Y) = minibatch
                # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                ### START CODE HERE ### (≈ 1 line of code)
                AL, caches = self.L_model_forward(minibatch_X, parameters)
                ### END CODE HERE ###
                
                # Compute cost.
                ### START CODE HERE ### (≈ 1 line of code)
                cost = self.compute_cost(AL, minibatch_Y, parameters, lambd)
                ### END CODE HERE ###
            
                # Backward propagation.
                ### START CODE HERE ### (≈ 1 line of code)
                grads = self.L_model_backward(AL, minibatch_Y, caches, lambd)
                ### END CODE HERE ###
         
                # Update parameters.
                ### START CODE HERE ### (≈ 1 line of code)
                parameters = self.update_parameters(parameters, grads, learning_rate)
                ### END CODE HERE ###
                        
            # Print the cost every 100 training example
            if print_cost and i % print_cost_interval == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % print_cost_interval == 0:
                costs.append(cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters
      
######################3
"""
Test
"""
def testOptimalMinibatchMnist():
    dnn = NNMnist()
    dnntc= NNMnistTestCase()
    
    mnist = dnntc.load_mnist_data()
    m,n = mnist.data.shape
    mnist.target = mnist.target.reshape(m,1)
    num_labels = 10
    e=np.eye(num_labels)

    #shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = mnist.data[permutation,:]
    shuffled_Y = mnist.target[permutation,:]
    
    shuffled_X = shuffled_X / 255
    
    
    train_x = shuffled_X[0:50000,:].T  #784x50000
    train_y =  shuffled_Y[0:50000,:].T #1x50000
    train_y = e[train_y[0,:].astype(int),:].T#10x50000
    
    dev_x = shuffled_X[50000:60000,:].T
    dev_y=  shuffled_Y[50000:60000,:].T
    dev_y = e[dev_y[0,:].astype(int),:].T#10x50000
   
    test_x = shuffled_X[60000:70000,:].T
    test_y=  shuffled_Y[60000:70000,:].T
    test_y = e[test_y[0,:].astype(int),:].T#10x50000
    
    
    print("train set shape")
    print(train_x.shape)
    print(train_y.shape)

    
    
    layers_dims =  [n, 30,  num_labels]#  5-layer model
    print("\n===================L_layer_model \n")
    parameters = dnn.L_layer_model(
        train_x, train_y, layers_dims,  print_cost = True, 
        lambd = 0, learning_rate = 0.1, 
        mini_batch_size=10, num_epochs=30,
        print_cost_interval=1)
     
    print("\n===================predict train\n")
    pred_train = dnn.predict(train_x, train_y, parameters)
     
    print("\n===================predict dev\n")
    pred_dev = dnn.predict(dev_x, dev_y, parameters)

def testComputeCost():
    dnn = NNMnist()
    dnntc= NNMnistTestCase()    
    print("\n===================compute_cost\n")
    Y, AL = dnntc.compute_cost_test_case()
    print("cost = expected: 0.414931599615, actual: " + str(dnn.compute_cost(AL, Y, [], 0)))
#     cost = 
def testMinibatch():
    dnn = NNMnist()
    dnntc= NNMnistTestCase()    
    X_assess, Y_assess, mini_batch_size = dnntc.random_mini_batches_test_case()
    mini_batches = dnn.random_mini_batches(X_assess, Y_assess, mini_batch_size)
    
    print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
    print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))
    
    
#     shape of the 1st mini_batch_X: (12288, 64)
#     shape of the 2nd mini_batch_X: (12288, 64)
#     shape of the 3rd mini_batch_X: (12288, 20)
#     shape of the 1st mini_batch_Y: (1, 64)
#     shape of the 2nd mini_batch_Y: (1, 64)
#     shape of the 3rd mini_batch_Y: (1, 20)
#     mini batch sanity check: [ 0.90085595 -0.7612069   0.2344157 ]
if __name__ == '__main__':
    testOptimalMinibatchMnist()
