import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - tanh(x) * tanh(x)

def logistic(x):
    return 1/(1+np.exp(-x))

def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))

#1: self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)  
#2: self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)

#In #1, layers[i-1] + 1 is to include bias so that it can be calculated using matrix dot product. Similarly in #2 layers[i] + 1 is for the same purpose. You can see input X is pre-processed to include additional one column with all values 1. 1*bias = bias 

#Layer #1's result will be used to calculate layer #2's value, in order to make the matrix calculable using dot product, number of columns of the first matrix must be the same as the second row's, I guess that's why he used layers[i] + 1 instead of layers[i]. However I think there is indeed a bug: After the hidden layer's output is calculated the last column of the matrix is not 1 anymore, it cannot be directly used to multiply the next layer's bias. It should be reset to 1 before calculating next layer's value.

def prt(x="",y=""):
    if 1== 1:
        print(x,y)
        
class NeuralNetwork:
    def __init__(self, layers, activation="tanh"):
        prt("===__init__===")
        if activation == "tanh":
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        elif activation == "logistic":
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        
    
        self.weights = []
        
        for i in range(1, len(layers) - 1):
            prt("No.",i)
            #Add one more row for bias
            weight = (2 * np.random.random((layers[i - 1] + 1, layers[i] +1)) - 1) * 0.25
            
            prt("weight1:"+str(i)+"\n"+str(weight))
            self.weights.append(weight)
            
            weight = (2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25
            prt("weight2:"+str(i)+"\n"+str(weight))
            self.weights.append(weight)
        ''' 
        weight=np.array([[0.2, -0.3, 0.1], 
                         [0.4, 0.1, 0.1], 
                         [-0.5, 0.2, 0.1],
                         [-0.4, 0.2, 0.1]
                         ])
        print(weight) 
        self.weights.append(weight)
        weight=np.array([[-0.3], 
                         [-0.2], 
                         [0.1]])
        self.weights.append(weight)
         '''
        prt("=== End of __init__ ===")
        prt()
        
    def fit(self,X,y,L=0.2,ephochs=10000):
        prt("===fit===")
        # Make sure it is at least a 2D matrix
        X=np.atleast_2d(X)
#         print("X:", X)
        # Create a temp matrix with all values 1
        temp = np.ones([X.shape[0], X.shape[1]+1])
#         print("temp:", temp)
        # Override all columns except for last one by X
        temp[:, 0:-1] = X   # adding the bias unit to the input layer
        
#         print("temp:", temp)
        X = temp # with one additional column with value 1 to calculate bias using dot
        
        prt("X:\n" +str(X))

        y = np.array(y)
        prt("y:\n" + str(y))


        for k in range(ephochs):
            #randomly pick one row
            i = np.random.randint(X.shape[0]) 
            prt("Processing No.",i)
            a = [X[i]]# 1x3 
            prt("a:\n",a)
            prt("Start forwarding calc")
            # forward calc
            for ly in range(len(self.weights)):#going forward network, for each layer
                prt("Layer "+str(ly))
                #print("a["+str(ly)+"]:\n"+str(a[ly]))
                #print("weights["+str(ly)+"]:\n"+str(self.weights[ly]))
                #a[ly][-1]=1;
                dotproduct = np.dot(a[ly],self.weights[ly]);# [](1x3) * [](3x3) = [] (1x3)
                act = self.activation(dotproduct);
                a.append(act) #Compute the node value for each layer (O_i) using activation function
                prt("self.activation(np.dot(a["+str(ly)+"],self.weights["+str(ly)+"])):\n "+str(a[ly])+" \nDOT\n"+ str(self.weights[ly])+"\n= "+str(dotproduct)+",\nOutput:\n"+str(act))
                
                prt(a)
                prt()
            # error
            error = y[i] - a[-1]; # top layer
            deltas  = [error * self.activation_deriv(a[-1])]#For output layer, Err calculation (delta is updated error)
            prt("a:\n"+str(error))
            prt("b:\n"+str(self.activation_deriv(a[-1])))
            prt("c:\n"+str(deltas))
            prt("===forward is done===")
            prt()
            
            
            #backpropagation
            for ly in range(len(a)-2,0,-1): # we need to begin at the second to last layer
                #Compute the updated error (i,e, deltas) for each node going from top layer to input layer 
                prt("self.weights[ly]\n"+str(self.weights[ly]))
                prt("self.weights[ly].T\n"+str(self.weights[ly].T))
                err_x_weight = deltas[-1].dot(self.weights[ly].T);
                prt(str(deltas[-1])+".dot("+str(self.weights[ly].T)+")="+str(err_x_weight))
                delta = err_x_weight*self.activation_deriv(a[ly]);
                prt(str(err_x_weight)+"*"+str(self.activation_deriv(a[ly]))+"="+str(delta))
                deltas.append(delta)
                prt("deltas["+str(ly)+"]:\n"+str(deltas))
            deltas.reverse()
            
            
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                prt("layer\n"+str(layer))
                delta = np.atleast_2d(deltas[i])
                prt("delta\n"+str(delta))
                dweight = L*layer.T.dot(delta)
                prt("dweight\n"+str(dweight))
                prt("self.weights[i]\n"+str(self.weights[i]))
                self.weights[i] += dweight
                prt("self.weights[i]\n"+str(self.weights[i]))
            prt("===back propagation is done===")
            
        prt("=== Fit is done ===")
    def predict(self, m):
        n = np.array(m)
       
        temp = np.ones(n.shape[0] + 1)
        temp[0:-1] = n
        a = temp
        for ly in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[ly]))
        return a



nn = NeuralNetwork([2,2,1], 'tanh')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])   
y = np.array([0, 1, 1, 0])    
 
nn.fit(X, y, ephochs=10000)
for i in [[0, 0], [0, 1], [1, 0], [1,1]]:    
    print(i, nn.predict(i))
'''
nn = NeuralNetwork([3,2,1], 'logistic')
X = np.array([[1, 0, 1]])  
y = np.array([1])
nn.fit(X, y, ephochs=1)
# print(nn.predict([0,20,1]))
'''
