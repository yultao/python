import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class UnivariateLinearRegression():

    def plotCost(self, J_history):
        plt.plot(J_history[:,0], '-', linewidth=2)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost J")
        plt.show()
    def plotData(self,X,y):
        plt.scatter(X[:,1], y, marker="x", color="red")
        plt.xlabel("Population of City in 10,000s")
        plt.ylabel("Profit in $10,000s")
        plt.show()
    
    def plotFunction(self,X,y):
        plt.scatter(X[:,1], y, marker="x", color="red")
        y_hat = ulr.predict(X, theta)
        plt.xlabel("Population of City in 10,000s")
        plt.ylabel("Profit in $10,00s")
        plt.plot(X[:,1], y_hat[:,0], '-')
        plt.show()
    def plotContour(self,theta):
        theta0_vals = np.linspace(-10, 10, 100);
        theta1_vals = np.linspace(-1, 4, 100);
        J_vals = np.zeros((len(theta0_vals), len(theta1_vals)));
        for i in range(0, len(theta0_vals)):
            for j in range(0, len(theta1_vals)):
                t = np.array([theta0_vals[i], theta1_vals[j]]).reshape(2,1);
                J_vals[i,j] = ulr.computeCost(X, y, t);
        J_vals = J_vals.T
        print(J_vals.shape)
        
        
        plt.contour(theta0_vals,theta1_vals,J_vals, np.logspace(-2, 3, 20))
        plt.plot(theta[0,0], theta[1,0],'x', color="red", linewidth=2);
        plt.show()
        
    def computeCost(self, X, y, theta):
        m = X.shape[0]
        J = np.sum((X.dot(theta) - y)**2)/(2*m);
        return J
    
    def gradientDescent(self, X, y, theta, alpha, num_iters):
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
    
    def leastSqured(self, X, y):
        m,n = X.shape
        X = X[:,1].reshape(m,1)
        xm = np.mean(X)
        ym = np.mean(y)
#          b_up = \sum{x_iy_i} - \sum{x_i}\sum{y_i}/n
#          b_down = \sum{x_ix_i} - \sum{x_i}\sum{x_i}/n
        up = np.sum(X*y) * m - np.sum(X) * np.sum(y);
        down = np.sum(X*X) * m - np.sum(X) * np.sum(X);

        b1 = up / down;
#         // a = \bar y - b \bar x
        b0 = ym - b1 * xm;
        theta = [b0, b1];
        return theta;
    
    def leastSqured2(self, X, y):
        m,n = X.shape
        X = X[:,1].reshape(m,1)
        xm = np.mean(X)
        ym = np.mean(y)

        up = np.sum( (X - xm)*(y-ym) );
        down = np.sum( (X - xm)*(X-xm) );

        b1 = up / down;
#         // a = \bar y - b \bar x
        b0 = ym - b1 * xm;
        theta = [b0, b1];
        return theta;
#     
    def normalEqn(self, X, y):
        theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return theta
if __name__ == '__main__':
    data = np.loadtxt("./data/ex1data1.txt", delimiter=',', skiprows=0)
    m = data.shape[0]
    X = np.ones((m,2))
    X[:,1] = data[:,0]
    y = np.zeros((m,1))
    y[:,0] = data[:,1]
    
    ulr = UnivariateLinearRegression()
    ulr.plotData(X, y)
    

    
    #Test cost
    theta = np.zeros((2,1))
    J = ulr.computeCost(X,y,theta)
    print("Expected: 32.07: "+str(J));
    
    #Test cost
    theta = np.array([-1,2]).reshape(2,1)
    J = ulr.computeCost(X,y,theta)
    print("Expected: 54.24: "+str(J));
    
    #Test gradientDescent
    alpha = 0.01
    num_iters = 1500
    theta = np.zeros((2,1))
    theta,J_history = ulr.gradientDescent(X, y, theta, alpha, num_iters)
    print("===========\nGradient descent 1500, Expected:\n  -3.6303\n  1.1664\n: Actual \n"+str(theta));
    
    alpha = 0.01
    num_iters = 15000
    theta = np.zeros((2,1))
    theta,J_history = ulr.gradientDescent(X, y, theta, alpha, num_iters)
    print("===========\nGradient descent 15000, Expected:\n  -3.89578088\n  1.19303364\n: Actual \n"+str(theta));
    
    
    theta_2 = ulr.leastSqured(X, y)
    print("===========\nleastSqured Expected:\n  -3.89578088\n  1.19303364\n: Actual \n"+str(theta_2));
    
    theta_3 = ulr.leastSqured2(X, y)
    print("===========\nleastSqured2 Expected:\n  -3.89578088\n  1.19303364\n: Actual \n"+str(theta_3));
    
    theta_4 = ulr.normalEqn(X, y)
    print("===========\nnormalEqn Expected:\n  -3.89578088\n  1.19303364\n: Actual \n"+str(theta_4));
    
    #Test predict
    predict1 = ulr.predict(np.array([1, 3.5]).reshape(1,2), theta);
    predict2 = ulr.predict(np.array([1, 7]).reshape(1,2), theta);
    print('For population = 35,000, we predict a profit of 4519.767868\n' + str( predict1*10000));
    print('For population = 70,000, we predict a profit of 45342.450129\n' + str( predict2*10000));
    
    #Plot cost
    ulr.plotCost(J_history)
    
    #Plot function
    ulr.plotFunction(X,y)
    
    #Plot contour
    ulr.plotContour(theta)
