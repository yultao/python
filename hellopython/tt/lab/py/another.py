import numpy as np


print("STATEMENTS in another starts")

e=np.eye(10)
print(e)
aa = e[:,2]
print(aa)


y=np.array([3,5]).reshape(2,1)

print(y)
print(y[:,0])
aa = e[y[:,0]-1,:]
print(aa)


y=np.array([0.1,0.3,0.4,0.2])
print(y.ravel())
idx = np.random.choice(4, 3, p = y.ravel())
print(idx)
print("STATEMENTS in another ends")

def whoami():
    print("I am another!")
if False:

    print(np.power(2,3))
    
    a_prev = np.zeros((2,3));
    print(a_prev);
    
    n_a,m=a_prev.shape
    xt = np.ones((3,3));
    n_x,m=xt.shape
    print(xt);
    
    #concat = np.concatenate((a_prev, xt))
    concat = np.zeros((n_a+n_x,m))
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = xt
    print(concat)
    
    
    print(n_a)
    print(concat[: n_a, :])
    
    print(concat[n_a :, :])

# if it is executed directly __name__ is '__main__';
# if it is imported by other modules, __name__ is 'another', which is the same as the file name
print("NAME in another: " + __name__)

if __name__ == '__main__':
    print("MAIN")
    pass