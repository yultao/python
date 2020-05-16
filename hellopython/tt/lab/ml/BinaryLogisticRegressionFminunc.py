import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

def plotData( data):
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
    

def sigmoid(z):
    return 1/(1 + np.exp(-z));

def gradientDescent(theta,x,y):
    m , n = x.shape
    theta = theta.reshape((n,1));
    y = y.reshape((m,1))
    sigmoid_x_theta = sigmoid(x.dot(theta));
    grad = ((x.T).dot(sigmoid_x_theta-y))/m;
    return grad.flatten();

def computeCost(theta,x,y):
    m,n = x.shape; 
    theta = theta.reshape((n,1));
    y = y.reshape((m,1));
    term1 = np.log(sigmoid(x.dot(theta)));
    term2 = np.log(1-sigmoid(x.dot(theta)));
    term1 = term1.reshape((m,1))
    term2 = term2.reshape((m,1))
    term = y * term1 + (1 - y) * term2;
    J = -((np.sum(term))/m);
    return J;
def prob( X, theta):
    y = sigmoid(X.dot(theta))
    return y
def predict( X, theta):
    y = sigmoid(X.dot(theta)) >= 0.5
    return y
if __name__ == '__main__':
    data = np.loadtxt("./data/ex2data1.txt", delimiter=',', skiprows=0)
    plotData(data)
    
    m = data.shape[0]
    n = data.shape[1]


    X = np.ones((m,n))
    X[:,1:n] = data[:,0:-1]
    y = np.zeros((m,1))
    y[:,0] = data[:,-1]
    

    initial_theta = np.array([-24, 0.2, 0.2])
    cost =computeCost(initial_theta,X, y);
    print('Expected cost (approx): 0.218\n actual: \n'+str(cost));
    grad = gradientDescent(initial_theta,X,y)
    print("expected: \n[[ 0.04290299]\n[ 2.56623412]\n [ 2.64679737]]\n actual: \n"+str(grad))
    initial_theta = np.zeros(n);
    Result = op.minimize(fun = computeCost,  x0 = initial_theta, 
                                 args = (X, y),
                                 method = 'TNC',
                                 jac = gradientDescent);
    optimal_theta = Result.x;
    print(optimal_theta)
    """
    test
    """
    """
    plot boundary
    """
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

    print(optimal_theta)
    plot_x = np.array([np.min(X[:,1])-2,  np.max(X[:,1])+2]).reshape(1,2);
    
    print(plot_x)#28.059   101.828

    plot_y = (-1/optimal_theta[2])*(optimal_theta[1]*plot_x + optimal_theta[0])
#     plot_y = plot_y_norm*sigma + mu
    print(plot_y)#96.166   20.653

    plt.plot(plot_x[0,:], plot_y[0,:], '-')
    
    plt.show()
    


