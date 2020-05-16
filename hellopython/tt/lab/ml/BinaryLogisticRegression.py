import numpy as np
import matplotlib.pyplot as plt


class BinaryLogisticRegression():
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
        
        plt.xlabel("Exam 1 score")
        plt.ylabel("Exam 2 score")
        plt.show()
    def plotCost(self, J_history):
        plt.plot(J_history[:,0], '-', linewidth=2)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost J")
        plt.show()
    def plotBoundary(self, data, theta):
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
        
        plt.xlabel("Exam 1 score")
        plt.ylabel("Exam 2 score")
    
        print(theta)
        plot_x = np.array([np.min(X[:,1])-2,  np.max(X[:,1])+2]).reshape(1,2);
        
        print(plot_x)#28.059   101.828
        plot_x_norm = (plot_x-mu[0,0]) / sigma[0,0];
        
        
        plot_y_norm = (-1/theta[2,0])*(theta[1,0]*plot_x_norm + theta[0,0])
        plot_y = plot_y_norm*sigma[0,0] + mu[0,0] #de-normalize
        print(plot_y)#96.166   20.653 func
    #     [[ 97.06288779  17.89804401]] correct
    #     [[ 97.06288779  19.49688718]] incorrect
        plt.plot(plot_x[0,:], plot_y[0,:], '-')
        
        plt.show()
    def sigmoid(self, z):
        return 1/( 1 + np.exp(-z));
    
    def computeCost(self, X, y, theta):
        m = y.shape[0];
        grad = np.zeros(theta.shape);
        h = self.sigmoid(X.dot(theta)); # mxn * nx1 = mx1
        #J = (1/m)*np.sum(np.multiply(-y, np.log(h)) - np.multiply((1-y), np.log(1-h)))
        J = (1/m) * np.sum((-y) * np.log(h) - (1-y) * np.log(1-h));             # mx1
        grad = ((1/m) * np.sum((h-y)*X, axis=0, keepdims=True)).T;           #  mx1 .* mxn => sum => 1xn => nx1
        return (J, grad)
    
    def featureNormalize(self, X):
        mu = np.mean(X,axis=0,keepdims=True);
        sigma = np.std(X,axis=0,keepdims=True);
        X_norm = (X-mu) / sigma;
        return (mu, sigma, X_norm)
    
    def gradientDescent(self, X, y, theta, alpha, num_iters):
        m = y.shape[0] 
        J_history = np.zeros((num_iters, 1))
        for iter in range(0, num_iters):
            grad = ((1/m) * np.sum((self.sigmoid(X.dot(theta))-y)*X, axis=0, keepdims=True)).T;
            #print(grad)
            theta = theta - grad;
            J_history[iter,0], a = self.computeCost(X, y, theta);
            #print(grad.shape)
            #print(theta)
        return (theta, J_history)
    def prob(self, X, theta):
        y = self.sigmoid(X.dot(theta))
        return y
    def predict(self, X, theta):
        y = self.sigmoid(X.dot(theta)) >= 0.5
        return y

if __name__ == '__main__':
    data = np.loadtxt("./data/ex2data1.txt", delimiter=',', skiprows=0)
    blr = BinaryLogisticRegression()
    blr.plotData(data)
    
    m = data.shape[0]
    n = data.shape[1]
    X = np.ones((m,n))
    X[:,1:n] = data[:,0:-1]
    y = np.zeros((m,1))
    y[:,0] = data[:,-1]
    
    test_theta = np.array([-24, 0.2, 0.2]).reshape(3,1);
    
    cost, grad = blr.computeCost(X, y,test_theta);
    print('Expected cost (approx): 0.218\n actual: \n'+str(cost));
    print("expected: \n[[ 0.04290299]\n[ 2.56623412]\n [ 2.64679737]]\n actual: \n"+str(grad))
    
    """
    test
    """
    
    X_orig = data[:,0:-1]
    mu, sigma, X_norm=blr.featureNormalize(X_orig)
    
    X_norm_1 = np.ones((m,n))
    X_norm_1[:,1:n+1] = X_norm
    
    iterations = 1000
    alpha = 0.01
    theta = np.zeros((n,1))
    theta,J_history = blr.gradientDescent(X_norm_1, y, theta, alpha, iterations)
    print(theta)
#     [[ 1.65947664]
#      [ 3.8670477 ]
#      [ 3.60347302]]
    blr.plotCost(J_history)
    """
    predict
    """
    xx = np.array([45,85]).reshape(1,2) 
    xx = (xx-mu) / sigma;
    x = np.ones((1,3))
    x[:,1:3] = xx[:,0:2]
    print(x.shape)
    prob =blr.prob(x,theta)
    print('For a student with scores 45 and 85, we predict an admission probability of 0.775 +/- 0.002\n' + str( prob));


    """
    """
    
    p = blr.predict(X_norm_1,theta)
#     print(p[0:10,:])
#     print("=========")
#     print(y[0:10,:])
#     print("=========")
#     print(p.shape)
    result = (p == y)
    accuracy = np.mean(result.astype(int))*100
    print("accuracy 89.0% "+ str(accuracy))
    
    """
    plot boundary
    """
    blr.plotBoundary(data,theta)