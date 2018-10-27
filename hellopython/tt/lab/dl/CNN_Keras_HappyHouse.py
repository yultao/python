import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


class CNNKeras():
    # GRADED FUNCTION: HappyModel

    def HappyModel(self, input_shape):
        """
        Implementation of the HappyModel.
        
        Arguments:
        input_shape -- shape of the images of the dataset
    
        Returns:
        model -- a Model() instance in Keras
        """
        
        ### START CODE HERE ###
        # Feel free to use the suggested outline in the text above to get started, and run through the whole
        # exercise (including the later portions of this notebook) once. The come back also try out other
        # network architectures as well. 
        
        # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
        X_input = Input(input_shape)
    
        # Zero-Padding: pads the border of X_input with zeroes
        X = ZeroPadding2D((3, 3))(X_input)
    
        # CONV -> BN -> RELU Block applied to X
        X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
        X = BatchNormalization(axis = 3, name = 'bn0')(X)
        X = Activation('relu')(X)
    
        # MAXPOOL
        X = MaxPooling2D((2, 2), name='max_pool')(X)
    
        # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
        X = Flatten()(X)
        X = Dense(1, activation='sigmoid', name='fc')(X)
    
        # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
        model = Model(inputs = X_input, outputs = X, name='HappyModel')
        
        ### END CODE HERE ###
        
        return model


class CNNKerasTestCase():
    def load_dataset(self):
        train_dataset = h5py.File('datasets/train_happy.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    
        test_dataset = h5py.File('datasets/test_happy.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    
        classes = np.array(test_dataset["list_classes"][:]) # the list of classes
        
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    
#     number of training examples = 600
#     number of test examples = 150
#     X_train shape: (600, 64, 64, 3)
#     Y_train shape: (600, 1)
#     X_test shape: (150, 64, 64, 3)
#     Y_test shape: (150, 1)
    def test_loaddata(self):
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = self.load_dataset()
        # Normalize image vectors
        X_train = X_train_orig/255.
        X_test = X_test_orig/255.
        
        # Reshape
        Y_train = Y_train_orig.T
        Y_test = Y_test_orig.T
        
        print ("number of training examples = " + str(X_train.shape[0]))
        print ("number of test examples = " + str(X_test.shape[0]))
        print ("X_train shape: " + str(X_train.shape))
        print ("Y_train shape: " + str(Y_train.shape))
        print ("X_test shape: " + str(X_test.shape))
        print ("Y_test shape: " + str(Y_test.shape))
        return (X_train, Y_train, X_test, Y_test, classes);
    def test_model(self, X_train, Y_train):
        cnn = CNNKeras()
        ### START CODE HERE ### (1 line)
        happyModel = cnn.HappyModel((X_train.shape[1],X_train.shape[2],X_train.shape[3]))
        ### END CODE HERE ###
        ### START CODE HERE ### (1 line)
        happyModel.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
        ### END CODE HERE ###
        ### START CODE HERE ### (1 line)
        happyModel.fit(x = X_train, y = Y_train, epochs = 40, batch_size = 16)
        ### END CODE HERE ###
        return happyModel;
    
    
#     150/150 [==============================] - 1s     
# 
#     Loss = 0.0836641703919
#     Test Accuracy = 0.960000003974
    def test_evaluate(self, happyModel, X_test, Y_test):
        ### START CODE HERE ### (1 line)
        preds = happyModel.evaluate(x = X_test, y = Y_test)
        ### END CODE HERE ###
        print()
        print ("Loss = " + str(preds[0]))
        print ("Test Accuracy = " + str(preds[1]))
        
    def test_own(self, happyModel):
        ### START CODE HERE ###
        img_path = 'images/happy.jpg'
        ### END CODE HERE ###
        img = image.load_img(img_path, target_size=(64, 64))
        imshow(img)
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        print(happyModel.predict(x))
        
if __name__ == '__main__':
    tc = CNNKerasTestCase();
    X_train, Y_train, X_test, Y_test, classes = tc.test_loaddata();
    happyModel = tc.test_model(X_train, Y_train);
    tc.test_evaluate(happyModel, X_test, Y_test);
    tc.test_own(happyModel);
    