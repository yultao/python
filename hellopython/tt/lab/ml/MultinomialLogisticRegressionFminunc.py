import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tt.lab.util import DataLoader as loader
import scipy.optimize as op

def featureNormalize(X):
    X_norm = X / 255;
    return X_norm    
def sigmoid(z):
    return 1/( 1 + np.exp(-z));
def plotCost(J_history):
    plt.plot(J_history[:,0], '-', linewidth=2)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost J")
    plt.show()
def computeCost22222222222222(X, y, theta, lambd):
    m,n = X.shape;
#         grad = np.zeros(theta.shape);
    h = sigmoid(X.dot(theta)); # mxn * nx1 = mx1
    #J = (1/m)*np.sum(np.multiply(-y, np.log(h)) - np.multiply((1-y), np.log(1-h)))
    J = (1/m) * np.sum((-y) * np.log(h) - (1-y) * np.log(1-h));             # mx1
    jr = (lambd/(2*m)) * np.sum( np.power( theta[1:n, 0], 2))
    J = J + jr
    
    grad = ((1/m) * np.sum((h-y)*X, axis=0, keepdims=True)).T;           #  mx1 .* mxn => sum => 1xn => nx1
    
    gr = (lambd/m)* theta[1:n,0]
    grad[1:n,0] = grad[1:n,0] + gr
    
    return (J, grad)
def computeCost(theta,x,y,lambd):
    m,n = x.shape; 
    theta = theta.reshape((n,1));
    y = y.reshape((m,1));
    term1 = np.log(sigmoid(x.dot(theta)));
    term2 = np.log(1-sigmoid(x.dot(theta)));
    term1 = term1.reshape((m,1))
    term2 = term2.reshape((m,1))
    term = y * term1 + (1 - y) * term2;
    J = -((np.sum(term))/m);
    jr = (lambd/(2*m)) * np.sum( np.power( theta[1:n, 0], 2))
    J = J + jr
    return J;
def gradientDescent(theta,x,y, lambd):
    m , n = x.shape
    theta = theta.reshape((n,1));
    y = y.reshape((m,1))
    sigmoid_x_theta = sigmoid(x.dot(theta));
    grad = ((x.T).dot(sigmoid_x_theta-y))/m;
    gr = (lambd/m)* theta[1:n,0]
    grad[1:n,0] = grad[1:n,0] + gr
    return grad.flatten();
def oneVsAll(X, y, num_labels, lambd):
    m,n = X.shape
    all_theta = np.zeros((num_labels, n));

    for c in range(0,num_labels):
        print("Label  "+str(c))
        initial_theta = np.zeros((n, 1));
        yc = y==c+1 # index 0 == number 1
        yc = yc.astype(int)
        
        Result = op.minimize(fun = computeCost,  x0 = initial_theta, 
                                     args = (X, yc, lambd),
                                     method = 'TNC',
                                     jac = gradientDescent);
        theta = Result.x;
        theta = theta.reshape(n,1)
        
        
        all_theta[c,:] = theta.T;
        
    return all_theta

def predictOneVsAll(all_theta, X):
    m,n= X.shape
    p = np.argmax(sigmoid(X.dot(all_theta.T)), axis=1) +1 # mxn * nxc = mxc 
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
    # Test case for lrCostFunction
    print('\nTesting lrCostFunction() with regularization');
    
    theta_t = np.array([-2, -1, 1, 2]).reshape(4,1);
    X_t = np.append(np.ones((5,1)), np.array(range(1,16)).reshape(3,5).T/10, axis=1);
    y_t = (np.array([1, 0, 1, 0, 1]).reshape(5,1) >= 0.5);
    y_t = y_t.astype(int);
    print(X_t)
    print(y_t)
    lambda_t = 3;
    J = computeCost(theta_t, X_t, y_t, lambda_t);
    
    print('\nCost: \n' + str(J));
    print('Expected cost: 2.534819\n');

    
    lambd = 0.1;
    num_labels = 10; 
#     X =featureNormalize(X)
    X = np.append(np.ones((m,1)), X, axis=1)
    
    all_theta = oneVsAll(X, y, num_labels, lambd);
    
    p = predictOneVsAll(all_theta, X);
    result = (p == y)
    accuracy = np.mean(result.astype(int))*100
    print('\nTraining Set Accuracy: 94.780000 \n' + str(accuracy));