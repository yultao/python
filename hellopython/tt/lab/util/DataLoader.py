import numpy as np
import scipy.io


def load(filename):
    print("Loading data: "+filename)
    data = np.loadtxt(filename, delimiter=',', skiprows=0)
    
    return data;

def loadMat(filename):
    print("Loading data: "+filename)
    data = scipy.io.loadmat(filename)
    
    return data;

def main():
    print("main")
    loadMat("../data/ex3data1.mat")
    
if __name__ == '__main__':
    print("run directly. __name__: "+__name__)
    main()
