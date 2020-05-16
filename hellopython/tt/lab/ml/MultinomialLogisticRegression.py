import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tt.lab.util import DataLoader as loader

''' In statistics, multinomial logistic regression is a classification method that generalizes logistic regression to multiclass problems, 
i.e. with more than two possible discrete outcomes.
'''

class MultinomialLogisticRegression():
    def featureNormalize(self, X):
        X_norm = X / 255;
        return X_norm    
    def sigmoid(self, z):
        return 1/( 1 + np.exp(-z));
    def plotCost(self, J_history):
        plt.plot(J_history[:,0], '-', linewidth=2)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost J")
        plt.show()
    def computeCost(self, X, y, theta, lambd):
        m,n = X.shape;
#         grad = np.zeros(theta.shape);
        h = self.sigmoid(X.dot(theta)); # mxn * nx1 = mx1
        #J = (1/m)*np.sum(np.multiply(-y, np.log(h)) - np.multiply((1-y), np.log(1-h)))
        J = (1/m) * np.sum((-y) * np.log(h) - (1-y) * np.log(1-h));             # mx1
        jr = (lambd/(2*m)) * np.sum( np.power( theta[1:n, 0], 2))
        J = J + jr
        
        grad = ((1/m) * np.sum((h-y)*X, axis=0, keepdims=True)).T;           #  mx1 .* mxn => sum => 1xn => nx1
        
        gr = (lambd/m)* theta[1:n,0]
        grad[1:n,0] = grad[1:n,0] + gr
        
        return (J, grad)
    def gradientDescent(self, X, y, theta, alpha, num_iters, lambd):
        m,n = X.shape;
        J_history = np.zeros((num_iters, 1))
        for iter in range(0, num_iters):
            grad = ((1/m) * np.sum((self.sigmoid(X.dot(theta))-y)*X, axis=0, keepdims=True)).T;
            
            gr = (lambd/m)* theta[1:n,0]
            grad[1:n,0] = grad[1:n,0] + gr
            

            theta = theta - grad;
            J_history[iter,0], a = self.computeCost(X, y, theta, lambd);

        return (theta, J_history)
    def oneVsAll(self, X, y, num_labels, lambd):
        m,n = X.shape
        all_theta = np.zeros((num_labels, n));
        alpha = 0.01
        num_iters = 1000
        
        for c in range(0,num_labels):
            print("Label  "+str(c))
            initial_theta = np.zeros((n, 1));
            yc = y==c+1 # index 0 == number 1
            yc = yc.astype(int)
            
            theta, J_history = self.gradientDescent(X, yc, initial_theta, alpha, num_iters, lambd)
#             if(c==0):
#                 self.plotCost(J_history)
            all_theta[c,:] = theta.T;
            
        return all_theta
    
    def predictOneVsAll(self, all_theta, X):
        m,n= X.shape
        p = np.argmax(self.sigmoid(X.dot(all_theta.T)), axis=1) +1 # mxn * nxc = mxc 
        p = p.reshape(m,1)
        return p
if __name__ == '__main__':
    data = loader.loadMat("./data/ex3data1.mat")
    X = data["X"]
    y = data["y"]
    print(X.shape)
    print(y.shape)
    m = X.shape[0]
    print(m)
    print()
    mlr = MultinomialLogisticRegression();
    # Test case for lrCostFunction
    print('\nTesting lrCostFunction() with regularization');
    
    theta_t = np.array([-2, -1, 1, 2]).reshape(4,1);
    X_t = np.append(np.ones((5,1)), np.array(range(1,16)).reshape(3,5).T/10, axis=1);
    y_t = (np.array([1, 0, 1, 0, 1]).reshape(5,1) >= 0.5);
    y_t = y_t.astype(int);
    print(X_t)
    print(y_t)
    lambda_t = 3;
    J, grad= mlr.computeCost(X_t, y_t, theta_t, lambda_t);
    
    print('\nCost: \n' + str(J));
    print('Expected cost: 2.534819\n');
    print('Gradients:\n');
    print(' %f \n'+ str(grad));
    print('Expected gradients:\n');
    print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');
    
    
    lambd = 0.1;
    num_labels = 10; 
#     X =featureNormalize(X)
    X = np.append(np.ones((m,1)), X, axis=1)
    all_theta = mlr.oneVsAll(X, y, num_labels, lambd);
    
    p = mlr.predictOneVsAll(all_theta, X);
    result = (p == y)
    accuracy = np.mean(result.astype(int))*100
    print('\nTraining Set Accuracy: 94.780000 \n' + str(accuracy));