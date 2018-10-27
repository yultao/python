
import numpy as np
L=3
for l in reversed(range(L-1)):
    print(l)
def onehot():
    num_labels=10
    e=np.eye(num_labels)
    print("eye:\n", e)
    test_y = np.array([3,1,9,0,4,8,9,2,1,0,7,6]).reshape(1,12)
    print("test_y:\n",  test_y)
    oh = e[:,test_y[0,:]]
    print("one-hot:\n", oh)
onehot()

def shuffle():# 
    m=5
    index = list(np.random.permutation(m))
    print("index:\n", index)
    
    #create a list from 0 to 4
    original = np.array(range(m)).reshape(1,m) #mx1
    print("original:\n", original)
    
    shuffled = original[:,index] #shuffle according to row
    print("shuffled:\n", shuffled)


shuffle()
def test3():
    a=np.random.randn(3,16)
    print(a)
    
    a=a.reshape(3,4,4, 1)
    print(a.shape)

def test2():
    a=np.array([2,1,0,22,112,1]).reshape((2,3))
    print(a)
    one = np.where(a == 1)
    print(one)
    print(a[one])

def test():
    e=np.eye(10)
    print(e)
    y=np.array([3,5]).reshape(2,1)
    
    print(y)
    print(y[:,0])
    aa = e[y[:,0]-1,:]
    print(aa)
    
    
    print("===============")
    e=np.eye(10)
    print(e)
    print(e.T)
    y=np.array([3,5,1,9,2,8,1,9,2,8,0]).reshape(1,11)
    
    print(y)
    print(y[0,:])
    aa = e[y[0,:],:]
    print(aa.T)
    
    bb = np.argmax(aa.T, axis=0).reshape(1,11)
    print(bb)