import matplotlib.pyplot as plt
import numpy as np
def plotCost(J_history):
    plt.plot(J_history[:,0], '-', linewidth=2)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost J")
    plt.show()
def plotData(data):
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
