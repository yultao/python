import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
from PIL import Image
from scipy import ndimage
from tensorflow.python.framework import ops
from sklearn.datasets import fetch_mldata
import os
from email._header_value_parser import Parameter
import pickle
from numpy import save
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CNNTFMnist():
    dropout = True
    
    def random_mini_batches(self, X, Y, mini_batch_size = 64, seed = 0):
        m = X.shape[0]                  # number of training examples
        mini_batches = []
        np.random.seed(seed)
        
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation,:,:,:]
        shuffled_Y = Y[permutation,:]
    
        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
            mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches
    
    # does not work
    def show_image(self, X_train_orig,Y_train_orig):
        # Example of a picture
        index = 6
        plt.imshow(X_train_orig[index])
        print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
        plt.show();
        
           
    def convert_to_one_hot(self, Y, C):
        Y = np.eye(C)[Y.reshape(-1)].T
        return Y
    # GRADED FUNCTION: create_placeholders

    def create_placeholders(self, n_H0, n_W0, n_C0, n_y):
        
        ### START CODE HERE ### (≈2 lines)
        X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
        Y = tf.placeholder(tf.float32, shape=(None, n_y))
        ### END CODE HERE ###
        
        return X, Y
    # GRADED FUNCTION: initialize_parameters
    
    def initialize_parameters(self):
        tf.set_random_seed(1)                              # so that your "random" numbers match ours
        ### START CODE HERE ### (approx. 2 lines of code)
#         W1 = tf.get_variable("W1", [3, 3, 1, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        W1 = tf.get_variable("W1", [5, 5, 1, 12], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
#         W2 = tf.get_variable("W2", [3, 3, 16, 32], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        W2 = tf.get_variable("W2", [3, 3, 12, 24], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

        ### END CODE HERE ###
        
        parameters = {"W1": W1,
                      "W2": W2}
        print("Init paramenters")
        return parameters
    
    def restore_parameters(self):
        W1 = tf.get_variable("W1", [5, 5, 1, 12])
        W2 = tf.get_variable("W2", [3, 3, 12, 24])
        parameters = {"W1": W1,
                      "W2": W2}
        return parameters
#         saver = tf.train.Saver()
#         with tf.Session() as sess:
#             # Restore variables from disk.
#             saver.restore(sess, "./models/cnn_tf_mnist_on_sign_parameters.ckpt")
#             print("Model restored.")
#             # Check the values of the variables
#             print("W1 = " + str(parameters["W1"].eval()[0,0,0]))
#             print("W2 = " + str(parameters["W2"].eval()[0,0,0]))
#             return parameters
#     def save_parameters(self, parameters):
# #         with open('./models/cnn_tf_mnist_on_sign_parameters.pkl', 'w') as file:
# # #             file.write(parameters)
# #             pickle.dump(parameters, file, pickle.HIGHEST_PROTOCOL)
#         print("dd", parameters)
#         np.save('./models/cnn_tf_mnist_on_sign_parameters.txt', parameters) 
#     def load_parameters(self):
# #         with open('./models/cnn_tf_mnist_on_sign_parameters.pkl', 'r') as file:
# #             parameters = pickle.load(file)
#         parameters = np.load('./models/cnn_tf_mnist_on_sign_parameters.txt').item()
#         return parameters    
            
    # GRADED FUNCTION: forward_propagation
    def forward_propagation(self, X, parameters):
        # Retrieve the parameters from the dictionary "parameters" 
        W1 = parameters['W1']
        W2 = parameters['W2']
        
        ### START CODE HERE ###
        # CONV2D: stride of 1, padding 'SAME'
#         Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
        Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'VALID')
        # RELU
        A1 = tf.nn.relu(Z1)
        # MAXPOOL: window 8x8, sride 8, padding 'SAME'
#         P1 = tf.nn.max_pool(A1, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
        P1 = tf.nn.max_pool(A1, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'VALID')
        # CONV2D: filters W2, stride 1, padding 'SAME'
#         Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
        Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
        # RELU
        A2 = tf.nn.relu(Z2)
        # MAXPOOL: window 4x4, stride 4, padding 'SAME'
#         P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
        P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
        # FLATTEN
        P2 = tf.contrib.layers.flatten(P2)
        # FULLY-CONNECTED without non-linear activation function (not not call softmax).
        # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
        Z3 = tf.contrib.layers.fully_connected(P2, 10 ,activation_fn=None)
        Z3 = tf.layers.dropout(Z3, rate=0.01, training=self.dropout)
        ### END CODE HERE ###
    
        return Z3  
    
    # GRADED FUNCTION: compute_cost 
    def compute_cost(self, Z3, Y):
        ### START CODE HERE ### (1 line of code)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
        ### END CODE HERE ###
        return cost

    # GRADED FUNCTION: model
    def model(self, X_train, Y_train, X_dev, Y_dev, learning_rate = 0.009,
              num_epochs = 100, minibatch_size = 64, print_cost = True, restore=False):
        
        ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
        seed = 3                                          # to keep results consistent (numpy seed)
        (m, n_H0, n_W0, n_C0) = X_train.shape             
        n_y = Y_train.shape[1]                            
        costs = []                                        # To keep track of the cost
        
        # Create Placeholders of the correct shape
        ### START CODE HERE ### (1 line)
        X, Y = self.create_placeholders(n_H0, n_W0, n_C0, n_y)
        ### END CODE HERE ###
    
 
        if restore==False :
            # Initialize parameters
            ### START CODE HERE ### (1 line)
            parameters = self.initialize_parameters()
            ### END CODE HERE ###
        else:
            parameters = self.restore_parameters()
        # Forward propagation: Build the forward propagation in the tensorflow graph
        ### START CODE HERE ### (1 line)
        Z3 = self.forward_propagation(X, parameters)
        ### END CODE HERE ###
        
        # Cost function: Add cost function to tensorflow graph
        ### START CODE HERE ### (1 line)
        cost = self.compute_cost(Z3, Y)
        ### END CODE HERE ###
        
        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
        ### START CODE HERE ### (1 line)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
        ### END CODE HERE ###
        
        # Initialize all the variables globally
        init = tf.global_variables_initializer()
        
        #save to disc
        saver = tf.train.Saver()
        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:

            # Run the initialization
            sess.run(init)
            
            if restore==True :
                # Restore variables from disk.
                saver.restore(sess, "./models/cnn_tf_mnist_on_sign_parameters.ckpt")
                print("Model restored.")
                # Check the values of the variables
                print("W1 = " + str(parameters["W1"].eval()[0,0,0]))
                print("W2 = " + str(parameters["W2"].eval()[0,0,0]))
            
            # Do the training loop
            for epoch in range(num_epochs):
    
                minibatch_cost = 0.
                num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = self.random_mini_batches(X_train, Y_train, minibatch_size, seed)
    
                for minibatch in minibatches:
    
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                    ### START CODE HERE ### (1 line)
                    _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    ### END CODE HERE ###
                    
                    minibatch_cost += temp_cost / num_minibatches
                    
    
                # Print the cost every epoch
                if print_cost == True and epoch % 1 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                if print_cost == True and epoch % 1 == 0:
                    costs.append(minibatch_cost)
                if print_cost == True and epoch % 5 == 0:
                    # Calculate the correct predictions
                    predict_op = tf.argmax(Z3, 1)
                    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                    
                    # Calculate accuracy on the test set
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #                 print(accuracy)
                    self.dropout = True
                    train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
                    print("Train Accuracy:", train_accuracy)
                    self.dropout = False
                    dev_accuracy = accuracy.eval({X: X_dev, Y: Y_dev})
                    print("Dev Accuracy:", dev_accuracy)
            
            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
#             plt.show()
    
            # Calculate the correct predictions
            predict_op = tf.argmax(Z3, 1)
            correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
            
            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#             print(accuracy)
            self.dropout = True
            train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
            print("Train Accuracy:", train_accuracy)
            self.dropout = False
            dev_accuracy = accuracy.eval({X: X_dev, Y: Y_dev})
            print("Dev Accuracy:", dev_accuracy)
        
            print("Training completes")    
            print("W1m = " + str(parameters["W1"].eval()[0,0,0]))
            print("W2m = " + str(parameters["W2"].eval()[0,0,0]))
#             save            
            save_path = saver.save(sess, "./models/cnn_tf_mnist_on_sign_parameters.ckpt")
            print("Model saved in path: %s" % save_path)
            return train_accuracy, dev_accuracy, parameters

## TFConvNN ##################
cnn = CNNTFMnist();

class CNNTFMnistTestCase():
    def load_mnist_data(self):
        mnist = fetch_mldata('MNIST original')
#         print(mnist.data.shape)
#         print(mnist.target.shape)

        m,n = mnist.data.shape
        mnist.target = mnist.target.reshape(m,1)
        num_labels = 10
        e=np.eye(num_labels)
    
        #shuffle
        permutation = list(np.random.permutation(m))
        shuffled_X = mnist.data[permutation,:]
        shuffled_Y = mnist.target[permutation,:]
        
#         shuffled_X = shuffled_X / 255
        
        
        train_x = shuffled_X[0:50000,:]  #784x50000
        train_y =  shuffled_Y[0:50000,:] #1x50000
#         train_y = e[train_y[0,:].astype(int),:].T#10x50000
#         train_y = cnn.convert_to_one_hot(train_y.astype(int), num_labels)
        
        dev_x = shuffled_X[50000:60000,:]
        dev_y=  shuffled_Y[50000:60000,:]
#         dev_y = e[dev_y[0,:].astype(int),:].T#10x50000
#         dev_y = cnn.convert_to_one_hot(dev_y.astype(int), num_labels)
        
        test_x = shuffled_X[60000:70000,:]
        test_y=  shuffled_Y[60000:70000,:]
#         test_y = e[test_y[0,:].astype(int),:].T#10x50000
#         test_y = cnn.convert_to_one_hot(test_y.astype(int), num_labels)
        train_x = train_x.reshape(50000, 28,28,1)
        dev_x = dev_x.reshape(10000, 28,28,1)
        test_x = test_x.reshape(10000, 28,28,1)
        
        return (train_x, train_y, dev_x, dev_y, test_x, test_y );
    

#     number of training examples = 1080
#     number of test examples = 120
#     X_train shape: (1080, 64, 64, 3)
#     Y_train shape: (1080, 6)
#     X_test shape: (120, 64, 64, 3)
#     Y_test shape: (120, 6)
    def process_data(self, X_train_orig,Y_train_orig,X_dev_orig, Y_dev_orig,X_test_orig, Y_test_orig):
        X_train = X_train_orig/255.
        X_dev = X_dev_orig/255.
        X_test = X_test_orig/255.
        Y_train = cnn.convert_to_one_hot(Y_train_orig.astype(int), 10).T
        Y_dev = cnn.convert_to_one_hot(Y_dev_orig.astype(int), 10).T
        Y_test = cnn.convert_to_one_hot(Y_test_orig.astype(int), 10).T
        print ("number of training examples = " + str(X_train.shape[0]))
        print ("number of test examples = " + str(X_test.shape[0]))
        print ("X_train shape: " + str(X_train.shape))
        print ("Y_train shape: " + str(Y_train.shape))
        print ("X_dev shape: " + str(X_dev.shape))
        print ("Y_dev shape: " + str(Y_dev.shape))
        print ("X_test shape: " + str(X_test.shape))
        print ("Y_test shape: " + str(Y_test.shape))
        return (X_train,Y_train,X_dev,Y_dev,X_test,Y_test)
#     X = Tensor("Placeholder:0", shape=(?, 64, 64, 3), dtype=float32)
#     Y = Tensor("Placeholder_1:0", shape=(?, 6), dtype=float32)
    def test_create_placeholders(self):
        X, Y = cnn.create_placeholders(64, 64, 3, 6)
        print ("X = " + str(X))
        print ("Y = " + str(Y))
        return (X,Y)
    
#     W1 = [ 0.00131723  0.14176141 -0.04434952  0.09197326  0.14984085 -0.03514394
#      -0.06847463  0.05245192]
#     W2 = [-0.08566415  0.17750949  0.11974221  0.16773748 -0.0830943  -0.08058
#      -0.00577033 -0.14643836  0.24162132 -0.05857408 -0.19055021  0.1345228
#      -0.22779644 -0.1601823  -0.16117483 -0.10286498]
    def test_initialize_parameters(self):
        tf.reset_default_graph()
        with tf.Session() as sess_test:
            parameters = cnn.initialize_parameters()
            init = tf.global_variables_initializer()
            sess_test.run(init)
            print("W1 = " + str(parameters["W1"].eval()[0,0,0]))
            print("W2 = " + str(parameters["W2"].eval()[0,0,0]))
            
#     def test_save_parameters(self): 
#         tf.reset_default_graph()
#         with tf.Session() as sess_test:
#             parameters = cnn.initialize_parameters()
#             init = tf.global_variables_initializer()
#             sess_test.run(init)
#             print("W1 = " + str(parameters["W1"].eval()[0,0,0]))
#             print("W2 = " + str(parameters["W2"].eval()[0,0,0]))
#         cnn.save_parameters(parameters)  
# 
#     def test_load_parameters(self): 
#         parameters = cnn.load_parameters()  
#         print("W1 = " + str(parameters["W1"].eval()[0,0,0]))
#         print("W2 = " + str(parameters["W2"].eval()[0,0,0]))

    def test_save_parameters_tf(self): 
        tf.reset_default_graph()
        parameters = cnn.initialize_parameters()
        saver = tf.train.Saver()
        with tf.Session() as sess_test:
            init = tf.global_variables_initializer()
            sess_test.run(init)
            print("W1 = " + str(parameters["W1"].eval()[0,0,0]))
            print("W2 = " + str(parameters["W2"].eval()[0,0,0]))
            save_path = saver.save(sess_test, "./models/test.ckpt")
            print("Model saved in path: %s" % save_path)
        
        
    def test_load_parameters_tf(self): 
        W1 = tf.get_variable("W1", [5, 5, 1, 12])
        W2 = tf.get_variable("W2", [3, 3, 12, 24])
        parameters = {"W1": W1,
                      "W2": W2}
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # Restore variables from disk.
            saver.restore(sess, "./models/test.ckpt")
            print("Model restored.")
            # Check the values of the variables
            print("W1 = " + str(parameters["W1"].eval()[0,0,0]))
            print("W2 = " + str(parameters["W2"].eval()[0,0,0]))
        
#     Z3 = [[ 1.44169843 -0.24909666  5.45049906 -0.26189619 -0.20669907  1.36546707]
#      [ 1.40708458 -0.02573211  5.08928013 -0.48669922 -0.40940708  1.26248586]]

#     Z3 = [[-0.44670227 -1.57208765 -1.53049231 -2.31013036 -1.29104376  0.46852064]
#      [-0.17601591 -1.57972014 -1.4737016  -2.61672091 -1.00810647  0.5747785 ]]
    def test_forward_propagation(self):
        tf.reset_default_graph()
        
        with tf.Session() as sess:
            np.random.seed(1)
            X, Y = cnn.create_placeholders(64, 64, 3, 6)
            parameters = cnn.initialize_parameters()
            Z3 = cnn.forward_propagation(X, parameters)
            init = tf.global_variables_initializer()
            sess.run(init)
            a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
            print("Z3 = " + str(a))
#     cost = 4.66487
#     cost = 2.91034
    def test_compute_cost(self):
        tf.reset_default_graph()
        
        with tf.Session() as sess:
            np.random.seed(1)
            X, Y = cnn.create_placeholders(64, 64, 3, 6)
            parameters = cnn.initialize_parameters()
            Z3 = cnn.forward_propagation(X, parameters)
            cost = cnn.compute_cost(Z3, Y)
            init = tf.global_variables_initializer()
            sess.run(init)
            a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
            print("cost = " + str(a))
    def test_model(self, X_train, Y_train, X_test, Y_test, learning_rate, minibatch_size, num_epochs, restore):
        cnn.restore = True
        _, _, parameters = cnn.model(X_train, Y_train, X_dev, Y_dev, 
                                     learning_rate = learning_rate, 
                                     minibatch_size = minibatch_size, 
                                     num_epochs= num_epochs, restore=restore)

    def test_show_image(self,X_train_orig, Y_train_orig):
        cnn.show_image(X_train_orig, Y_train_orig)
## TFConvNNTestCase ##################
if __name__ == '__main__':
    tc = CNNTFMnistTestCase();
    
#     tc.test_create_placeholders();
#     tc.test_initialize_parameters();
#     tc.test_save_parameters_tf();
#     tc.test_load_parameters_tf()
#     tc.test_forward_propagation();
#     tc.test_compute_cost();
    
#     X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tc.load_dataset()
    X_train_orig, Y_train_orig, X_dev_orig, Y_dev_orig, X_test_orig, Y_test_orig = tc.load_mnist_data()
#     tc.test_show_image(X_train_orig,Y_train_orig);
    X_train,Y_train,X_dev,Y_dev,X_test,Y_test = tc.process_data(X_train_orig,Y_train_orig,
                                                   X_dev_orig, Y_dev_orig, 
                                                   X_test_orig, Y_test_orig)

    for lr in [0.001]:
        for mb in [100]:
            print("learning rate: ", lr, ", mini-batch size:  ", mb)
            tc.test_model(X_train, Y_train, X_dev, Y_dev,learning_rate = lr,num_epochs = 5, minibatch_size = mb, restore=True);
            
# number of training examples = 50000
# number of test examples = 10000
# X_train shape: (50000, 28, 28, 1)
# Y_train shape: (50000, 10)
# X_dev shape: (10000, 28, 28, 1)
# Y_dev shape: (10000, 10)
# X_test shape: (10000, 28, 28, 1)
# Y_test shape: (10000, 10)
# learning rate:  0.001 , mini-batch size:   100
# Cost after epoch 0: 0.502915
# Dev Accuracy: 0.955
# Cost after epoch 1: 0.129893
# Cost after epoch 2: 0.096914
# Cost after epoch 3: 0.081308
# Dev Accuracy: 0.9748
# Cost after epoch 4: 0.069839
# Cost after epoch 5: 0.062978
# Cost after epoch 6: 0.056836
# Dev Accuracy: 0.9817
# Cost after epoch 7: 0.051910
# Cost after epoch 8: 0.047160
# Cost after epoch 9: 0.045463
# Dev Accuracy: 0.9859
# Cost after epoch 10: 0.041984
# Cost after epoch 11: 0.038908
# Cost after epoch 12: 0.037877
# Dev Accuracy: 0.9835
# Cost after epoch 13: 0.035365
# Cost after epoch 14: 0.033023
# Cost after epoch 15: 0.031015
# Dev Accuracy: 0.9867
# Cost after epoch 16: 0.030210
# Cost after epoch 17: 0.028816
# Cost after epoch 18: 0.027768
# Dev Accuracy: 0.9853
# Cost after epoch 19: 0.025401
# Cost after epoch 20: 0.024310
# Cost after epoch 21: 0.023851
# Dev Accuracy: 0.9881
# Cost after epoch 22: 0.022908
# Cost after epoch 23: 0.021706
# Cost after epoch 24: 0.020689
# Dev Accuracy: 0.987
# Cost after epoch 25: 0.019015
# Cost after epoch 26: 0.019137
# Cost after epoch 27: 0.019541
# Dev Accuracy: 0.9861
# Cost after epoch 28: 0.017011
# Cost after epoch 29: 0.016742
# Cost after epoch 30: 0.015640
# Dev Accuracy: 0.988
# Cost after epoch 31: 0.015199
# Cost after epoch 32: 0.014810
# Cost after epoch 33: 0.014536
# Dev Accuracy: 0.9881
# Cost after epoch 34: 0.013541
# Cost after epoch 35: 0.012826
# Cost after epoch 36: 0.013052
# Dev Accuracy: 0.989
# Cost after epoch 37: 0.011760
# Cost after epoch 38: 0.011126
# Cost after epoch 39: 0.010802
# Dev Accuracy: 0.9884
# Cost after epoch 40: 0.009878
# Cost after epoch 41: 0.009965
# Cost after epoch 42: 0.010623
# Dev Accuracy: 0.9875
# Cost after epoch 43: 0.008708
# Cost after epoch 44: 0.008569
# Cost after epoch 45: 0.009561
# Dev Accuracy: 0.988
# Cost after epoch 46: 0.007052
# Cost after epoch 47: 0.008543
# Cost after epoch 48: 0.007388
# Dev Accuracy: 0.9871
# Cost after epoch 49: 0.006886
# Cost after epoch 50: 0.007276
# Cost after epoch 51: 0.007711
# Dev Accuracy: 0.9891
# Cost after epoch 52: 0.006410
# Cost after epoch 53: 0.006022
# Cost after epoch 54: 0.006118
# Dev Accuracy: 0.9872
# Cost after epoch 55: 0.005289
# Cost after epoch 56: 0.005214
# Cost after epoch 57: 0.004852
# Dev Accuracy: 0.9873
# Cost after epoch 58: 0.006122
# Cost after epoch 59: 0.005987
# Train Accuracy: 0.99952
# Dev Accuracy: 0.9892


# ====
# W1m = [ 0.06833707 -0.36522642  0.01245116  0.23777473  0.25077015  0.09650254
#  -0.34448507  0.05485127  0.25417519  0.19344734  0.13012014  0.22413278]
# W2m = [  7.51292612e-03   1.08589903e-01   9.16838944e-02   1.43193349e-01
#   -1.01510957e-01  -1.19715601e-01   6.51468476e-03   1.62912271e-04
#   -1.98757462e-02   1.84565663e-01  -1.30378276e-01  -2.25797556e-02
#    8.56796466e-03  -2.27459788e-01  -1.95982724e-01  -1.37631431e-01
#   -3.24723683e-02  -6.40183017e-02  -7.37043992e-02  -7.94705600e-02
#   -6.75427169e-02   1.29715875e-01   1.35166079e-01  -6.83814958e-02]
# Model saved in path: ./models/cnn_tf_mnist_on_sign_parameters.ckpt

# Model restored.
# W1 = [ 0.06833707 -0.36522642  0.01245116  0.23777473  0.25077015  0.09650254
#  -0.34448507  0.05485127  0.25417519  0.19344734  0.13012014  0.22413278]
# W2 = [  7.51292612e-03   1.08589903e-01   9.16838944e-02   1.43193349e-01
#   -1.01510957e-01  -1.19715601e-01   6.51468476e-03   1.62912271e-04
#   -1.98757462e-02   1.84565663e-01  -1.30378276e-01  -2.25797556e-02
#    8.56796466e-03  -2.27459788e-01  -1.95982724e-01  -1.37631431e-01
#   -3.24723683e-02  -6.40183017e-02  -7.37043992e-02  -7.94705600e-02
#   -6.75427169e-02   1.29715875e-01   1.35166079e-01  -6.83814958e-02]
# Cost after epoch 0: 0.540057
# Train Accuracy: 0.95054
# Dev Accuracy: 0.9509
# Cost after epoch 1: 0.133229
# Cost after epoch 2: 0.097469
# Train Accuracy: 0.975
# Dev Accuracy: 0.9739
# Training completes
# W1m = [ 0.02405857 -0.32971549  0.08309232  0.28184995  0.22805655  0.0442479
#  -0.33523947  0.01082777  0.24593331  0.16153426  0.11825995  0.21785948]
# W2m = [-0.00105636  0.11502985  0.06939614  0.14160909 -0.0875255  -0.11579047
#   0.00172322 -0.04148162 -0.00892567  0.16449659 -0.13181557 -0.0075577
#   0.04072258 -0.18951394 -0.22353996 -0.14846054 -0.00323398 -0.04481377
#  -0.06713121 -0.07634232 -0.00706189  0.14626333  0.16752306 -0.05759329]
# Model saved in path: ./models/cnn_tf_mnist_on_sign_parameters.ckpt

# 
# Model restored.
# W1 = [ 0.02405857 -0.32971549  0.08309232  0.28184995  0.22805655  0.0442479
#  -0.33523947  0.01082777  0.24593331  0.16153426  0.11825995  0.21785948]
# W2 = [-0.00105636  0.11502985  0.06939614  0.14160909 -0.0875255  -0.11579047
#   0.00172322 -0.04148162 -0.00892567  0.16449659 -0.13181557 -0.0075577
#   0.04072258 -0.18951394 -0.22353996 -0.14846054 -0.00323398 -0.04481377
#  -0.06713121 -0.07634232 -0.00706189  0.14626333  0.16752306 -0.05759329]
# Cost after epoch 0: 0.080887
# Train Accuracy: 0.97508
# Dev Accuracy: 0.9723
# Cost after epoch 1: 0.071187
# Cost after epoch 2: 0.063672
# Train Accuracy: 0.97778
# Dev Accuracy: 0.9751
# Training completes
# W1m = [ 0.03048443 -0.36397976  0.07044674  0.34659672  0.22249337  0.05498696
#  -0.43829688 -0.00714921  0.30738157  0.158337    0.18356495  0.24152209]
# W2m = [ 0.00689176  0.12486793  0.05830203  0.09325355 -0.14249279 -0.14853306
#  -0.00507243 -0.07236656 -0.01648258  0.18531357 -0.12073257 -0.01800057
#   0.03509541 -0.18871677 -0.218018   -0.19040051  0.07744434 -0.0414533
#  -0.05418498 -0.11601773  0.00686665  0.16773571  0.18876791 -0.05266478]
# Model saved in path: ./models/cnn_tf_mnist_on_sign_parameters.ckpt