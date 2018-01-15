import numpy as np
import scipy.io


class NeuralNetworkPredictOnly():
    def sigmoid(self,z):
        return 1/(1 + np.exp(-z));    
    def predict(self, X, Theta1, Theta2):
        print(Theta1.shape) # r x n+1
        print(Theta2.shape)
#         a1 = [ones(m, 1) X]';
        m,n=X.shape
        
        a1 = np.append(np.ones((m,1)), X, axis = 1); # m x n+1
        
#         a2 = sigmoid(Theta1*a1);
        a2 = self.sigmoid(Theta1.dot(a1.T)) # r x n+1  * n+1 x m = r x m
        a2 = a2.T
#         a2 = [ones(1,m); a2];
        a2 = np.append(np.ones((m,1)), a2, axis = 1)
        
        a3 = self.sigmoid(Theta2.dot(a2.T));
        
        a3 = a3.T
        p = np.argmax(a3, axis=1) + 1;
        p = p.reshape(m,1)
        return p
    
if __name__ == '__main__':
    data = scipy.io.loadmat("../data/ex3data1.mat")
    X = data["X"]
    y = data["y"]
    weights= scipy.io.loadmat("../data/ex3weights.mat")
    Theta1 = weights["Theta1"]
    Theta2 = weights["Theta2"]
    nn = NeuralNetworkPredictOnly()
    p = nn.predict(X, Theta1, Theta2)
    
    result = (p == y)
    accuracy = np.mean(result.astype(int))*100
    print('\nTraining Set Accuracy: 97.520000 \n' + str(accuracy));