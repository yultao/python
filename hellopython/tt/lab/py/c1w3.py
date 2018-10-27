import numpy as np
import math
m=15
mini_batch_size=4
num_complete_minibatches=math.floor(m/mini_batch_size)
print(num_complete_minibatches)
last = m%mini_batch_size

for k in range(0, num_complete_minibatches):
    print("k "+str(k*mini_batch_size)+"~"+str((k+1)*mini_batch_size))
    
print(num_complete_minibatches*mini_batch_size)
def ff():
    A = np.random.randn(4,3)
    B = np.sum(A, axis = 1, keepdims = True)
    print(B.shape)
    
    a = np.array([1,2,3,4,5])
    b = (a>3)
    print(b)
    
    
    layer_dims = [5, 4,3,2,1]
    parameter={}
    for i in range(1, len(layer_dims)):
        parameter["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
        parameter["b" + str(i)] = np.random.randn(layer_dims[i], 1) * 0.01
    print("W1")
    print(parameter["W1"].shape)
    print(parameter["b1"].shape)
    
    print("W2")
    print(parameter["W2"].shape)
    print(parameter["b2"].shape)
    
    print("W3")
    print(parameter["W3"].shape)
    print(parameter["b3"].shape)
    
    print("W4")
    print(parameter["W4"].shape)
    print(parameter["b4"].shape)
    
    
    xx=0
    for i in range(1,10):
        xx=i
        print(xx)
        
    print(xx)
    
    bb = np.random.rand(3,1)
