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
    
def mapFeature(X1, X2):
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
def sigmoid(z):
    return 1/(1 + np.exp(-z));

def gradientDescent(theta,x,y, lambd):
    m , n = x.shape
    theta = theta.reshape((n,1));
    y = y.reshape((m,1))
    sigmoid_x_theta = sigmoid(x.dot(theta));
    grad = ((x.T).dot(sigmoid_x_theta-y))/m;
    gr = (lambd/m)* theta[1:n,0]
    grad[1:n,0] = grad[1:n,0] + gr
    return grad.flatten();

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
def prob( X, theta):
    y = sigmoid(X.dot(theta))
    return y
def predict( X, theta):
    y = sigmoid(X.dot(theta)) >= 0.5
    return y

def plotBoundary( data, optimized_theta):
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
            z[i,j] = np.dot(mapFeature(u[i].reshape(1,1), v[j].reshape(1,1)), optimized_theta);
    z = z.T
    print(z.shape)
    plt.contour(u, v, z, [0])
    plt.show()
if __name__ == '__main__':
    data = np.loadtxt("./data/ex2data2.txt", delimiter=',', skiprows=0)
    plotData(data)
    
    m = data.shape[0]
    n = data.shape[1]


    X = data[:, 0:-1].reshape(m,n-1)
    X = mapFeature(X[:,0], X[:,1])
    
    y = np.zeros((m,1))
    y[:,0] = data[:,-1]
    
    
    n = X.shape[1]
    
    """
    test cost with zero theta
    """
    print("=============")
    print("test cost with zero theta")
    initial_theta = np.zeros(n);
    lambd = 1
    cost = computeCost(initial_theta, X, y,lambd);
    print('Cost at initial theta (zeros): \n'+ str(cost));
    print('Expected cost (approx): 0.693\n');
    
    """
    test cost with one theta
    """
    print("=============")
    print("test cost with one theta")
    test_theta = np.ones(X.shape[1]);
    cost=computeCost(test_theta, X, y,10);
    print('\nCost at test theta (with lambda = 10): \n'+ str(cost));
    print('Expected cost (approx): 3.16\n');

    
    """
    test gradient descent
    """
    print("=============")
    print("test gradient descent")
    lambd=1
    initial_theta = np.zeros(n);
    Result = op.minimize(fun = computeCost,  x0 = initial_theta, 
                                 args = (X, y, lambd),
                                 method = 'TNC',
                                 jac = gradientDescent);
    optimal_theta = Result.x;
    print(optimal_theta)
    
    """
    predict
    """
    print("=============")
    print("predict")
    p = predict(X, optimal_theta)
    result = (p == y)
    accuracy = np.mean(result.astype(int))*100
    print("Expected accuracy (with lambda = 1): 83.1 (approx)\n"+ str(accuracy))

    """
    plot boundary
    """
    print("=============")
    print("plot boundary")
    plotBoundary(data, optimal_theta)

