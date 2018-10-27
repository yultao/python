import numpy as np
import matplotlib.pyplot as plt

class UnivariatePolynomialRegression(object):
    def plotData(self,data):
        m=data.shape[0]
        X = np.ones((m,2))
        X[:,1] = data[:,0]
        y = np.zeros((m,1))
        y[:,0] = data[:,1]
        plt.scatter(X[:,1], y, marker="x", color="blue")
        plt.xlabel("Population of City in 10,000s")
        plt.ylabel("Profit in $10,000s")
        plt.show()
    def plotCost(self, J_history):
        plt.plot(J_history[:,0], '-', linewidth=2)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost J")
        plt.show()
    def plotFunction(self, X, theta):
        plt.scatter(X[:,1], y, marker="x", color="blue")
        y_hat = self.predict(X, theta)
        plt.xlabel("Population of City in 10,000s")
        plt.ylabel("Profit in $10,00s")
        plt.plot(X[:,1], y_hat[:,0], '-', color="red")
        plt.show()
        
    def computeCost(self, X, y, theta):
        m = X.shape[0]
        J = np.sum((X.dot(theta) - y)**2)/(2*m);
        return J
    
    def mapFeature(self, X1, X2):
        degree = 2;
        m = X1.shape[0]
        X1 = X1.reshape(m,1)
        X2 = X2.reshape(m,1)
        out = np.ones((m,1));
        for i in range(1,degree+1):
            for j in range(0,i+1):
                z= np.power(X1,(i-j))  * np.power(X2, j).reshape(m,1);
                out = np.append(out, z, axis=1)
        return out;
    def featureNormalize(self, X):
        mu = np.mean(X,axis=0,keepdims=True);
        sigma = np.std(X,axis=0,keepdims=True);
        X_norm = (X-mu) / sigma;
        return (mu, sigma, X_norm)
    def gradientDescent(self, X, y, theta, alpha, num_iters, lambd):
        m = X.shape[0]
        J_history = np.zeros((num_iters, 1))
        for iterate in range(0, num_iters):
            grad = np.sum((X.dot(theta) - y)*X, axis=0,keepdims=True)*alpha/m;
            theta = theta - grad.T;
            J_history[iterate,0] = self.computeCost(X, y, theta);
        return (theta, J_history)

    def predict(self, X, theta):
        y = X.dot(theta)
        return y
    
if __name__ == '__main__':
    data = np.loadtxt("./data/ex1data3.txt", delimiter=',', skiprows=0)
    upr = UnivariatePolynomialRegression()
    upr.plotData(data)
    
    m,n = data.shape
    X = data[:, 0].reshape(m,1)
    #X = upr.mapFeature(X[:,0], X[:,0])


    X = np.append(X, np.power(X,2), axis=1)
    mu, sigma, X=upr.featureNormalize(X)
    X = np.append(np.ones((m,1)), X, axis=1)

    print(X[0:5,:])
    y = data[:, 1].reshape(m,1)
    print(X.shape)
    
    """
    test gradient descent
    """
    print("=============")
    print("test gradient descent")
    initial_theta = np.zeros((X.shape[1], 1));
    lambd = 0
    iterations = 5000
    alpha = 0.01
    optimized_theta,J_history = upr.gradientDescent( X, y, initial_theta, alpha, iterations, lambd)
    print(optimized_theta)
#     [[ 34.03810199]
#      [ 13.15137836]
#      [ 16.68703802]]

    upr.plotCost(J_history)
    
    upr.plotFunction(X,optimized_theta)
    
  

