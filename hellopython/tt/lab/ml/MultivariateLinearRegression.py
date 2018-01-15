import numpy as np
import matplotlib.pyplot as plt


class MultivariateLinearRegression():
    
    def featureNormalize(self, X):
        mu = np.mean(X,axis=0,keepdims=True);
        sigma = np.std(X,axis=0,keepdims=True);
        X_norm = (X-mu) / sigma;
        return (mu, sigma, X_norm)
    
    def computeCost(self, X, y, theta):
        m = y.shape[0] 
        J = np.sum((X.dot(theta) - y)**2)/(2*m);
        return J
    
    def gradientDescent(self, X, y, theta, alpha, num_iters):
        m = y.shape[0] 
        J_history = np.zeros((num_iters, 1))
        for it in range(0, num_iters):
            nabla = np.sum((X.dot(theta) - y)*X, axis=0,keepdims=True)*alpha/m;
            theta = theta - nabla.T;
            J_history[it,0] = self.computeCost(X, y, theta);
        return (theta, J_history)
    
    def normalEqn(self, X, y):
        theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return theta
    def plotCost(self, J_history):
        plt.plot(J_history[:,0], '-', linewidth=2)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost J")
        plt.show()
if __name__ == '__main__':
    
    mlr = MultivariateLinearRegression()
    data = np.loadtxt("../data/ex1data2.txt", delimiter=',', skiprows=0)
    m = data.shape[0]
    n = data.shape[1]
    
    X_orig = data[:,0:n-1]
    mu, sigma, X_norm=mlr.featureNormalize(X_orig)
    X = np.ones((m,n))
    X[:,1:n] = X_norm
    
    y = np.zeros((m,1))
    y[:,0] = data[:,-1]
    

    
    alpha = 0.01
    num_iters = 400
    
    
    #Test gradientDescent
    theta = np.zeros((3, 1))
    theta,J_history = mlr.gradientDescent(X, y, theta, alpha, num_iters)
    print("Theta calculated via gradient descent \n[[ 334302.06399328]\n [  99411.44947359]\n [   3267.01285407]]: \n" + str(theta))
    
    #Plot cost
    mlr.plotCost(J_history)
    
    
    # predict with normalization
    X = np.array([1650,3])
    X_norm = (X-mu) / sigma;
    X = np.ones((1,3)) 
    X[:,1:3]= X_norm[:,0:2];
    price = X.dot(theta);
    print("Price calculated via gradient descent: [[ 289221.54737122]]\n" + str(price))
    
    
    
    """
    Normal equation
    """
    
    # Reset X, no normalization
    X = np.ones((m, data.shape[1]))
    X[:,1:n] = data[:,0:n-1]
    y = np.zeros((m,1))
    
    y[:,0] = data[:,-1]
    theta = mlr.normalEqn(X, y)
    print("Theta calculated via normal equation: \n[[ 89597.9095428 ]\n [   139.21067402]\n [ -8738.01911233]]\n" + str(theta))
    
    # predict with normalization
    X = np.array([1, 1650,3])
    price = X.dot(theta);
    print("Price calculated via gradient descent: [ 293081.4643349]\n" + str(price))
