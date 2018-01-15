import numpy as np
import scipy.io

class NeuralNetwork():
    def sigmoid(self,z):
        return 1/(1 + np.exp(-z));
    def sigmoidGradient(self,z):
        return self.sigmoid(z) * (1-self.sigmoid(z))
    def randInitializeWeights(self, L_in, L_out):
        epsilon_init = 0.12;
        W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
        return W
    def computeCost(self,Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambd):

        m,n = X.shape

        # 400 - 25 - 10
        #Part 1
        #Theta1                                                     # 25x401
        #Theta2                                                     # 10x26
        a1 = np.append(np.ones((m, 1)), X, axis=1);                 # 5000x401
        z2 = a1.dot(Theta1.T);                                      # 5000x401 * 401*25 = 5000x25
        a2 = np.append(np.ones((m, 1)), self.sigmoid(z2), axis=1);  # 5000x26
        z3 = a2.dot(Theta2.T);                                      # 5000x26 * 26x10 = 5000x10
        a3 = self.sigmoid(z3);                                      # 5000x10
        
        y_matrix = np.eye(num_labels)[y[:,0]-1,:]

        J = (1/m) * np.sum(np.sum(-y_matrix * np.log(a3) - (1-y_matrix) * np.log(1-a3))) 
        jr = (lambd/(2*m)) * ( np.sum(np.sum ( np.power(Theta1[:,1:Theta1.shape[1]], 2) )) + np.sum(np.sum(np.power(Theta2[:,1:Theta1.shape[1]],2))) );
        J = J + jr
          
        #Part 2 
        
        Delta1 = np.zeros(Theta1.shape);       # 25x401
        Delta2 = np.zeros(Theta2.shape);       # 10x26
        
        #vectorize super fast!
        d3 = a3-y_matrix;                                           # 5000x10
        g2 = self.sigmoidGradient(z2);                # 5000x25
        d2 = d3.dot( Theta2[:,1:Theta2.shape[1]]) * g2;                            # 5000x10 * 10x25 .* 5000x25 = 5000x25
        
        Delta1 = d2.T.dot(a1);                                      # 25x5000 * 5000x401 = 25x401
        Delta2 = d3.T.dot(a2);                                      # 10x5000 * 5000x26  = 10x26
        '''
        #for-loop slow
        #{
        
        for t=1:m
          # forward
          a1 = [1 X(t,:)]';                 # 401x1
          z2 = Theta1 * a1;                 # 25x401 * 401x1 = 25x1
          a2 = [1; sigmoid(z2)];            # 26x1
          z3 = Theta2 * a2;                 # 10x26 * 26x1 = 10x1
          a3 = sigmoid(z3);                 # 10x1
          
          # back
          y = y_matrix(t,:)';               # 10x1
          d3 = a3 - y;                      # 10x1  error is respected to unit
          
          gp = sigmoidGradient(z2);         # 25x1
          d2 = Theta2' * d3 .* [0; gp];     # 26x10 * 10x1 .* 26x1 = 26x1
          d2 = d2(2:end);                   # 25x1  error is respectected to unit
          
          Delta2 = Delta2 + d3 * a2';       # 10x26  + (10x1 * 1x26)  = 10x26     nabla is respected to weight
          Delta1 = Delta1 + d2 * a1';       # 25x401 + (25x1 * 1x401) = 25x401    nabla is respected to weight
        endfor
        
        #}
        '''

        Theta1_grad = Delta1/m;             # 25x401
        Theta2_grad = Delta2/m;             # 10x26
        
        # -------------------------------------------------------------
        
        #Part 3
        
        Theta1_grad = Theta1_grad + (lambd/m)* np.append(np.zeros((Theta1.shape[0],1)), Theta1[:, 1:Theta1.shape[1]], axis=1); 
        Theta2_grad = Theta2_grad + (lambd/m)* np.append(np.zeros((Theta2.shape[0],1)), Theta2[:, 1:Theta2.shape[1]], axis=1); 
        # =========================================================================
        
        # Unroll gradients
        grad={}
        grad["Theta1_grad"]= Theta1_grad
        grad["Theta2_grad"]=Theta2_grad

        return (J, grad)

    def predict(self, X, Theta1, Theta2):
        #print(Theta1.shape) # r x n+1
        #print(Theta2.shape) # s x r+1

        m,n=X.shape
        
        a1 = np.append(np.ones((m,1)), X, axis = 1); # m x n+1
        

        a2 = self.sigmoid(Theta1.dot(a1.T)) # r x n+1  * n+1 x m = r x m
        a2 = a2.T

        a2 = np.append(np.ones((m,1)), a2, axis = 1)
        
        a3 = self.sigmoid(Theta2.dot(a2.T));
        a3 = a3.T
        
        
        p = np.argmax(a3, axis=1) + 1;
        p = p.reshape(m,1)
        return p
    
if __name__ == '__main__':
    """
    initial data
    """
    data = scipy.io.loadmat("../data/ex4data1.mat")
    X = data["X"]
    y = data["y"]
    weights= scipy.io.loadmat("../data/ex4weights.mat")

    Theta1 = weights["Theta1"]
    Theta2 = weights["Theta2"]
    
    input_layer_size  = 400;  # 20x20 Input Images of Digits
    hidden_layer_size = 25;   # 25 hidden units
    num_labels = 10;          # 10 labels, from 1 to 10   
    
    
    nn = NeuralNetwork()
    
    """
    Test cost lambd = 0;
    """
    lambd = 0;
    J, grad = nn.computeCost(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)
    print('\nCost at parameters (loaded from ex4weights): \n(this value should be about 0.287629)\n' + str(J));
    
    """
    Test cost lambd = 1;
    """
    lambd = 1;
    J, grad = nn.computeCost(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)
    print('\nCost at parameters (loaded from ex4weights): \n(this value should be about 0.383770)\n' + str(J));
    
    g = nn.sigmoidGradient(np.array([-1,-0.5,0,0.5,1]));
    print('\nSigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:' + str(g));
    
    

    
    """
    Test cost lambd = 3;
    """
    lambd = 3;
    debug_J, grad = nn.computeCost(Theta1, Theta2,  input_layer_size, hidden_layer_size, num_labels, X, y, lambd);

    print('\nCost at (fixed) debugging parameters (w/ lambda = 3):\n(for lambda = 3, this value should be about 0.576051)\n' + str(debug_J));
    
    
    initial_Theta1 = nn.randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = nn.randInitializeWeights(hidden_layer_size, num_labels);
    
    """
    Test predict
    """
    p = nn.predict(X, Theta1, Theta2)
    
    result = (p == y)
    accuracy = np.mean(result.astype(int))*100
    print('\nTraining Set Accuracy: 97.520000 \n' + str(accuracy));