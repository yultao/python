import numpy as np
import matplotlib.pyplot as plt

class BinaryLogisticRegressionRegularized():
    def plotData(self, data):
        p_data = data[data[:,-1]==1,:]
        n_data = data[data[:,-1]==0,:]
    
        pm = p_data.shape[0]
        nm = n_data.shape[0]
    
        p_X = p_data[:,0:-1]
        p_y = np.zeros((pm,1))
        p_y[:,0] = p_data[:,-1]
        plt.scatter(p_X[:,0], p_X[:,1], marker="+", color="red")
        
        
        n_X = n_data[:,0:-1]
        n_y = np.zeros((nm,1))
        n_y[:,0] = n_data[:,-1]
        plt.scatter(n_X[:,0], n_X[:,1], marker="o", color="blue")
        
        plt.xlabel("Microchip Test 1")
        plt.ylabel("Microchip Test 2")
        plt.show()
    def plotCost(self, J_history):
        plt.plot(J_history[:,0], '-', linewidth=2)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost J")
        plt.show()
        
    def plotBoundary(self, data, optimized_theta):
        p_data = data[data[:,-1]==1,:]
        n_data = data[data[:,-1]==0,:]
    
        pm = p_data.shape[0]
        nm = n_data.shape[0]
    
        p_X = p_data[:,0:-1]
        p_y = np.zeros((pm,1))
        p_y[:,0] = p_data[:,-1]
        plt.scatter(p_X[:,0], p_X[:,1], marker="+", color="red")
        
        
        n_X = n_data[:,0:-1]
        n_y = np.zeros((nm,1))
        n_y[:,0] = n_data[:,-1]
        plt.scatter(n_X[:,0], n_X[:,1], marker="o", color="blue")
        
        plt.xlabel("Microchip Test 1")
        plt.ylabel("Microchip Test 2")
        
        ############
        u = np.linspace(-1, 1.5, 50)#.reshape((50,1));
        print(u.shape)
        v = np.linspace(-1, 1.5, 50)#.reshape((50,1));
    
        z = np.zeros((len(u), len(v)));
        for i in range(0,len(u)):
            for j in range(0,len(v)):
                z[i,j] = np.dot(blrr.mapFeature(u[i].reshape(1,1), v[j].reshape(1,1)), optimized_theta);
        z = z.T
        print(z.shape)
        plt.contour(u, v, z, [0])
        plt.show()
        
    def mapFeature(self, X1, X2):
        degree = 6;
        m = X1.shape[0]
        X1 = X1.reshape(m,1)
        X2 = X2.reshape(m,1)
        out = np.ones((m,1));
        for i in range(1,degree+1):
            for j in range(0,i+1):
                z= np.power(X1,(i-j))  * np.power(X2, j).reshape(m,1);
                out = np.append(out, z, axis=1)
        return out;
    
    def sigmoid(self, z):
        return 1/( 1 + np.exp(-z));
    
    
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
    
    def featureNormalize(self, X):
        mu = np.mean(X,axis=0,keepdims=True);
        sigma = np.std(X,axis=0,keepdims=True);
        X_norm = (X-mu) / sigma;
        return (mu, sigma, X_norm)
    
    def gradientDescent(self, X, y, theta, alpha, num_iters, lambd):
        m,n = X.shape;
        J_history = np.zeros((num_iters, 1))
        for iter in range(0, num_iters):
            grad = ((1/m) * np.sum((self.sigmoid(X.dot(theta))-y)*X, axis=0, keepdims=True)).T;
            
            gr = (lambd/m)* theta[1:n,0]
            grad[1:n,0] = grad[1:n,0] + gr
            
            #print(grad)
            theta = theta - grad;
            J_history[iter,0], a = self.computeCost(X, y, theta, lambd);
            #print(grad.shape)
            #print(theta)
        return (theta, J_history)

    def predict(self, X, theta):
        y = self.sigmoid(X.dot(theta)) >= 0.5
        return y
if __name__ == '__main__':
    data = np.loadtxt("./data/ex2data2.txt", delimiter=',', skiprows=0)
    blrr = BinaryLogisticRegressionRegularized()
    blrr.plotData(data)
    
    m,n = data.shape
    X = data[:, 0:-1].reshape(m,n-1)
    X = blrr.mapFeature(X[:,0], X[:,1])
    y = data[:, -1].reshape(m,1)
    
    """
    test cost with zero theta
    """
    print("=============")
    print("test cost with zero theta")
    initial_theta = np.zeros((X.shape[1], 1));
    lambd = 1
    cost, grad = blrr.computeCost(X, y,initial_theta, lambd);
    print('Cost at initial theta (zeros): \n'+ str(cost));
    print('Expected cost (approx): 0.693\n');
    print('Gradient at initial theta (zeros) - first five values only:\n');
    print(grad[0:5]);
    print('Expected gradients (approx) - first five values only:\n');
    print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');
    
    """
    test cost with one theta
    """
    print("=============")
    print("test cost with one theta")
    test_theta = np.ones((X.shape[1], 1));
    cost, grad = blrr.computeCost(X, y, test_theta, 10);
    print('\nCost at test theta (with lambda = 10): \n'+ str(cost));
    print('Expected cost (approx): 3.16\n');
    print('Gradient at test theta - first five values only:\n');
    print(grad[0:5]);
    print('Expected gradients (approx) - first five values only:\n');
    print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');
    
    
    """
    test gradient descent
    """
    print("=============")
    print("test gradient descent")
    initial_theta = np.zeros((X.shape[1], 1));
    lambd = 0
    iterations = 5000
    alpha = 0.001
    optimized_theta,J_history = blrr.gradientDescent( X, y, initial_theta, alpha, iterations, lambd)
    print(optimized_theta)
    blrr.plotCost(J_history)
    
    
    """
    predict
    """
    print("=============")
    print("predict")
    p = blrr.predict(X, optimized_theta)
    result = (p == y)
    accuracy = np.mean(result.astype(int))*100
    print("Expected accuracy (with lambda = 1): 83.1 (approx)\n"+ str(accuracy))

    """
    plot boundary
    """
    print("=============")
    print("plot boundary")
    blrr.plotBoundary(data, optimized_theta)