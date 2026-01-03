import numpy as np
from scipy.special import softmax, logsumexp

class Network(object):
    
    def __init__(self, sizes):
        """
        The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        is [784, 40, 10] then it would be a three-layer network, with the
        first layer (the input layer) containing 784 neurons, the second layer 40 neurons,
        and the third layer (the output layer) 10 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution centered around 0.
        """
        self.num_layers = len(sizes) - 1
        self.sizes = sizes
        self.parameters = {}
        for l in range(1, len(sizes)):
            self.parameters['W' + str(l)] = np.random.randn(sizes[l], sizes[l-1]) * np.sqrt(2. / sizes[l-1])
            self.parameters['b' + str(l)] = np.zeros((sizes[l], 1))


    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)


    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
        


    def cross_entropy_loss(self, logits, y_true):
        m = y_true.shape[0]
        # Compute log-sum-exp across each column for normalization
        log_probs = logits - logsumexp(logits, axis=0)
        y_one_hot = np.eye(10)[y_true].T  # Assuming 10 classes
        # Compute the cross-entropy loss
        loss = -np.sum(y_one_hot * log_probs) / m
        return loss

    def cross_entropy_derivative(self, logits, y_true):
        """
        Input: logits (v_L), y_true (indices)
        Returns: Gradient dL/dv_L
        """
        # 1. Use SciPy to get probabilities (Handles stability automatically)
        #    axis=0 ensures we normalize down the column (per image)
        probs = softmax(logits, axis=0)
        
        # 2. Compute Gradient (Probs - Indicator of True Class)
        grad = probs.copy()
        batch_size = logits.shape[1]
        
        # Subtract 1 from the correct class index for each example
        grad[y_true, np.arange(batch_size)] -= 1
        
        return grad
    

    def linear_forward(self, A, W, b):
        """
        Helper 1: Pure Linear Calculation
        Z = WA + b
        """
        return np.dot(W, A) + b

    
    def layer_forward(self, A_prev, W, b, activation="relu"):
        """
        Helper to perform one layer's forward pass:
        Z = W * A_prev + b
        Output = Activation(Z)
        """
        # 1. Linear Step
        Z = self.linear_forward(A_prev, W, b)
        
        # 2. Activation Step
        if activation == "relu":
            return self.relu(Z)
        elif activation == "softmax":
            return softmax(Z, axis=0)
        else:
            # "" (linear) - used for the last layer before Softmax
            return Z
        


    def forward_propagation(self, X):
        """
        Implement the forward step of the backpropagation algorithm.
        Input: "X" - numpy array of shape (784, batch_size) - the input to the network
        Returns: "ZL" - numpy array of shape (10, batch_size), the output of the network on the input X (before the softmax layer)
        "forward_outputs" - A list of length self.num_layers containing the forward computation (parameters & output of each layer).
        
        """
        # 1. Initialize cache with input
        forward_outputs = [X]
        A_prev = X
        
        # 2. Hidden Layers Loop (1 to L-1)
        for i in range(1, self.num_layers):
            # Retrieve parameters (Updated Name)
            W = self.parameters['W' + str(i)]
            b = self.parameters['b' + str(i)]
            
            # Apply Linear -> ReLU using the helper
            A_prev = self.layer_forward(A_prev, W, b, activation="relu")
            
            # Store activation
            forward_outputs.append(A_prev)

        # 3. Output Layer (Layer L)
        L = self.num_layers
        # Retrieve parameters (Updated Name)
        W_last = self.parameters['W' + str(L)]
        b_last = self.parameters['b' + str(L)]
        
        # Use "identity" so we get raw Logits (before softmax)
        ZL = self.layer_forward(A_prev, W_last, b_last, activation="identity")
        
        forward_outputs.append(ZL)

        return ZL, forward_outputs

    

    def calculate_output_error(self, ZL, Y):
        """
        Optimized: Avoids creating a separate One-Hot matrix.
        """
        m = Y.shape[0]
        
        # 1. Calculate Probabilities
        P = softmax(ZL, axis=0)
        
        # 2. Calculate Error in-place
        #    Copy P to dZ so we don't modify the cache if we needed P later (we don't here, but it's safe)
        dZ = P
        
        #    Instead of (P - Y_one_hot), we just find the correct label indices and subtract 1
        #    This is O(m) instead of O(10*m) matrix subtraction
        dZ[Y, np.arange(m)] -= 1
        
        return dZ


    def linear_backward(self, dZ, A_prev):
        """
        Computes gradients dW and db for a specific layer.
        """
        m = A_prev.shape[1]
        
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        return dW, db
    
    
    def relu_backward(self, dZ_next, W_next, A_curr):
        """
        Optimized: Uses boolean indexing to avoid multiplying by a mask matrix.
        """
        # 1. Pull error back (Linear part)
        #    dZ_curr = W.T * dZ_next
        dZ_curr = np.dot(W_next.T, dZ_next)
        
        # 2. ReLU Derivative (The Optimization)
        #    Instead of: dZ = dZ * (A > 0)
        #    We do: "Where A is 0, set dZ to 0"
        #    This avoids allocating a mask matrix and doing element-wise multiplication.
        dZ_curr[A_curr <= 0] = 0
        
        return dZ_curr




    def backpropagation(self, ZL, Y, forward_outputs):
        """Implement the backward step of the backpropagation algorithm.
            Input: "ZL" -  numpy array of shape (10, batch_size), the output of the network on the input X (before the softmax layer)
                    "Y" - numpy array of shape (batch_size,) containing the labels of each example in the current batch.
                    "forward_outputs" - list of length self.num_layers given by the output of the forward function
            Returns: "grads" - dictionary containing the gradients of the loss with respect to the network parameters across the batch.
                                grads["dW" + str(l)] is a numpy array of shape (sizes[l], sizes[l-1]),
                                grads["db" + str(l)] is a numpy array of shape (sizes[l],1).
        
        """

        grads = {}
        L = self.num_layers
        
        # --- STEP 1: Output Layer (Layer L) ---
        # Get the initial error
        dZ = self.calculate_output_error(ZL, Y)
        
        # Get gradients for the last layer
        # forward_outputs[-2] is the input to the last layer (A_{L-1})
        A_prev = forward_outputs[-2]
        
        # Calculate & Store Gradients
        grads["dW" + str(L)], grads["db" + str(L)] = self.linear_backward(dZ, A_prev)
        
        # --- STEP 2: Hidden Layers Loop (Backwards) ---
        # Loop from L-1 down to 1
        for l in range(L - 1, 0, -1):
            
            # Retrieve parameters needed for propagation
            W_next = self.parameters["W" + str(l + 1)]
            A_curr = forward_outputs[l]
            
            # 1. Propagate Error Backwards (Get dZ for this layer)
            dZ = self.relu_backward(dZ, W_next, A_curr)
            
            # 2. Calculate Gradients for this layer
            A_prev = forward_outputs[l-1]
            grads["dW" + str(l)], grads["db" + str(l)] = self.linear_backward(dZ, A_prev)
            
        return grads



    def sgd_step(self, grads, learning_rate):
        """
        Updates the network parameters via SGD with the given gradients and learning rate.
        """
        parameters = self.parameters
        L = self.num_layers
        for l in range(L):
            parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
        return parameters

    def train(self, x_train, y_train, epochs, batch_size, learning_rate, x_test, y_test):
        epoch_train_cost = []
        epoch_test_cost = []
        epoch_train_acc = []
        epoch_test_acc = []
        for epoch in range(epochs):
            costs = []
            acc = []
            for i in range(0, x_train.shape[1], batch_size):
                X_batch = x_train[:, i:i+batch_size]
                Y_batch = y_train[i:i+batch_size]

                ZL, caches = self.forward_propagation(X_batch)
                cost = self.cross_entropy_loss(ZL, Y_batch)
                costs.append(cost)
                grads = self.backpropagation(ZL, Y_batch, caches)

                self.parameters = self.sgd_step(grads, learning_rate)

                preds = np.argmax(ZL, axis=0)
                train_acc = self.calculate_accuracy(preds, Y_batch, batch_size)
                acc.append(train_acc)

            average_train_cost = np.mean(costs)
            average_train_acc = np.mean(acc)
            print(f"Epoch: {epoch + 1}, Training loss: {average_train_cost:.20f}, Training accuracy: {average_train_acc:.20f}")

            epoch_train_cost.append(average_train_cost)
            epoch_train_acc.append(average_train_acc)

            # Evaluate test error
            ZL, caches = self.forward_propagation(x_test)
            test_cost = self.cross_entropy_loss(ZL, y_test)
            preds = np.argmax(ZL, axis=0)
            test_acc = self.calculate_accuracy(preds, y_test, len(y_test))
            # print(f"Epoch: {epoch + 1}, Test loss: {test_cost:.20f}, Test accuracy: {test_acc:.20f}")

            epoch_test_cost.append(test_cost)
            epoch_test_acc.append(test_acc)

        return self.parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc


    def calculate_accuracy(self, y_pred, y_true, batch_size):
      """Returns the average accuracy of the prediction over the batch """
      return np.sum(y_pred == y_true) / batch_size
