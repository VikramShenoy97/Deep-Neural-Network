import numpy as np

class NeuralNetwork():
    """
        Deep Neural Network

        Parameters
        ----------
        layers_dims : array, shape[number_of_hidden_layers]
        Specifies number of nodes in each hidden layer

        learning_rate : int, Optional (default  = 0.0075)
        Rate at which the algorithm learns.

        num_iterations : int, Optional (default = 2000)
        Number of iterations for training.

        verbose : boolean, Optional (default=False)
        Controls verbosity of output:
        - False: No Output
        - True: Displays the cost at every 1000th iteration.

        regularization : boolean, Optional (default=None)
        Adds regularization if set to:
        - L2 : Adds L2 regularization effect

        lambd : int, Optional (default=0.1)
        Regularization parameter lambda, affects the neural network only if
        regularization is set to "L2".

        Attributes
        ----------
        costs_ : array, shape=[number_of_iterations/100]
        Returns an array with the costs.

        parameters_ : dictionary, {"W1":W1, "b1":b1, "W2":W2, "b2":b2 ....}
        Returns the weights and bias.

    """
    def __init__(self, layers_dims, learning_rate=0.0075, num_iterations=2000, verbose=False, regularization=None, lambd=0.1):
        self.layers_dims = layers_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.regularization = regularization
        self.lambd = lambd

    def fit(self, X, Y):
        """
        Fits the neural network on to the data and labels.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        Y : array-like, shape = [n_samples]
            The target values.

		Returns
        -------
        self : object

        """
        # If more than two layers (Deep Network)
        if(len(self.layers_dims) > 2):
            return self._L_layer_model(X, Y)
        # If only two layers (Shallow Network)
        else:
            return self._two_layer_model(X,Y)

    def predict(self, X):
        """
        Predicts the outcome of the given (data, label) pair.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

		Returns
        -------
        predictions: array, shape = [n_samples]
            Prediction of the test data.
        """
        return self._predict(X)

    def evaluate_loss(self):
        """
        Stores the cost at every 100th iteration.

        Parameters
        ----------
        self : object

		Returns
        -------
        costs: array, shape=[number_of_iterations/10]
        Returns an array with the costs.
        """
        return self._evaluate_loss()

    def accuracy(self, predictions, Y):
        """
        The Accuracy of the predicted label and the actual label.

        Parameters
        ----------
        predictions : array-like, shape = [n_samples]
            The predicted values.
        Y : array-like, shape = [n_samples]
            The target values.

		Returns
        -------
        Accuracy: float
        Accuracy of the predicted values.
        """
        return self._accuracy(predictions, Y)

    def _sigmoid(self, Z):
        """
        Perform the sigmoid activation function for forward propagation.
        """
        s = 1. / (1 + np.exp(-Z))
        cache = Z
        return s, cache

    def _relu(self, Z):
        """
        Perform the ReLU activation function for forward propagation.
        """
        r = np.maximum(0, Z)
        assert(r.shape == Z.shape)
        cache = Z
        return r, cache

    def _sigmoid_backward(self, dA, cache):
        """
        Perform sigmoid activation function for backward propagation.
        (Derivative of Sigmoid Activation Function)
        """
        Z = cache
        s = 1. / (1+np.exp(-Z))
        dZ = dA * s * (1-s)
        assert(dZ.shape == Z.shape)
        return dZ

    def _relu_backward(self, dA, cache):
        """
        Perform the ReLU activation function for backward propagation.
        (Derivative of ReLU Activation Function)
        """
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert(dZ.shape == Z.shape)
        return dZ

    def _initialize_parameters(self, n_x, n_h, n_y):
        """
        Initialize the weights and the bias for Shallow Network.
        - Weights are randomly initialized.
        - Bias are initialized as zeros.
        """
        np.random.seed(1)
        parameters = {}

        W1 = np.random.randn(n_h, n_x)*0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h)*0.01
        b2 = np.zeros((n_y, 1))
        # Check the Shape of weights and bias.
        assert(W1.shape == (n_h, n_x))
        assert(b1.shape == (n_h, 1))
        assert(W2.shape == (n_y, n_h))
        assert(b2.shape == (n_y, 1))

        parameters = {"W1": W1, "b1":b1, "W2":W2, "b2":b2}
        return parameters

    def _initialize_parameters_deep(self):
        """
        Initialize the weights and the bias for Deep Network.
        - Weights are initialized using Xavier Initialization.
        - Bias are initialized as zeros.
        """
        np.random.seed(2)
        L = len(self.layers_dims)
        parameters = {}
        for l in range(1, L):
            parameters["W"+str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l-1]) / np.sqrt(self.layers_dims[l-1])
            parameters["b"+str(l)] = np.zeros((self.layers_dims[l], 1))
            # Check the Shape of weights and bias.
            assert(parameters["W"+str(l)].shape == (self.layers_dims[l], self.layers_dims[l-1]))
            assert(parameters["b"+str(l)].shape == (self.layers_dims[l], 1))
        return parameters

    def _linear_forward(self, A, W, b):
        """
        Forward Propagation to calculate the value of Z.
        """
        Z = np.dot(W, A) + b
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        return Z, cache

    def _linear_activation_forward(self, A_prev, W, b, activation):
        """
        Forward Propagation to calculate the activation value A.
        """
        # For the last Layer.
        if activation == "sigmoid":
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A, activation_cache = self._sigmoid(Z)
        # For L-1 Layers
        elif activation == "relu":
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A, activation_cache = self._relu(Z)
        # Check the shape of the Activation A.
        assert(A.shape == (W.shape[0],A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
        return A, cache

    def _L_model_forward(self, X, parameters):
        """
        Performs complete forward propagation.
        """
        L = len(parameters) // 2
        A = X
        caches = []
        for l in range(1, L):
            A_prev = A
            A, cache = self._linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], activation="relu")
            caches.append(cache)
        AL, cache = self._linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], activation="sigmoid")
        caches.append(cache)
        # Checks the shape of final activation.
        assert(AL.shape == (1, X.shape[1]))
        return AL, caches

    def _compute_cost(self, AL, Y, parameters):
        """
        Calculates the overall cost of the network.
        """
        m = Y.shape[1]
        cost = (-1. / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))
        # If Regularization is added, the cost changes accordingly.
        if(self.regularization == "L2"):
            regularization_penalty = 0
            for l in range(1, (len(parameters)//2)+1):
                # L2 Loss
                store_weight = np.sum(np.square(parameters["W"+str(l)]))
                regularization_penalty = regularization_penalty + store_weight
            cost = cost + ((self.lambd / (2 * m)) * regularization_penalty)
        np.squeeze(cost)
        assert(cost.shape == ())
        return cost

    def _linear_backward(self, dZ, cache):
        """
        Backward Propagation to compute the linear derivatives.
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1. / m) * np.dot(dZ, A_prev.T)
        db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert(dW.shape == W.shape)
        assert(db.shape == b.shape)
        assert(dA_prev.shape == A_prev.shape)

        return dA_prev, dW, db

    def _linear_activation_backward(self, dA, cache, activation):
        """
        Backward Propagation to compute the linear derivatives through the
        derivatives of Activation functions.
        """
        linear_cache, activation_cache = cache
        if activation == "sigmoid":
            dZ = self._sigmoid_backward(dA, activation_cache)

        elif activation=="relu":
            dZ = self._relu_backward(dA, activation_cache)

        dA_prev, dW, db = self._linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def _L_model_backward(self, AL, Y, caches, parameters):
        """
        Performs complete backward propagation and computes the gradients.
        """
        gradients = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)
        m = Y.shape[1]
        dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
        assert(dAL.shape == Y.shape)
        current_cache = caches[-1]
        gradients["dA"+str(L)], gradients["dW"+str(L)], gradients["db"+str(L)] = self._linear_activation_backward(dAL, current_cache, activation="sigmoid")
        if(self.regularization == "L2"):
            gradients["dW"+str(L)] = gradients["dW"+str(L)] + (self.lambd / m)* parameters["W"+str(L)]

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_temp, dW_temp, db_temp = self._linear_activation_backward(gradients["dA"+str(l+2)], current_cache, activation="relu")
            if(self.regularization == "L2"):
                dW_temp = dW_temp + (self.lambd / m)* parameters["W"+str(l+1)]
            gradients["dA"+str(l+1)] = dA_temp
            gradients["dW"+str(l+1)] = dW_temp
            gradients["db"+str(l+1)] = db_temp

        return gradients

    def _update_parameters(self, parameters, gradients):
        """
        Optimize the parameters using the gradients.
        """
        L = len(parameters) // 2
        for l in range(L):
            parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - self.learning_rate * gradients["dW"+str(l+1)]
            parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - self.learning_rate * gradients["db"+str(l+1)]

        return parameters

    def _two_layer_model(self, X, Y):
        """
        Run the Shallow Neural Network.
        """
        np.random.seed(1)
        gradients = {}
        costs = []
        m = X.shape[1]
        n_x, n_h, n_y = self.layers_dims
        parameters = self._initialize_parameters(n_x, n_h, n_y)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        for i in range(self.num_iterations):

            A1, cache1 = self._linear_activation_forward(X, W1, b1, activation="relu")
            A2, cache2 = self._linear_activation_forward(A1, W2, b2, activation="sigmoid")

            cost = self._compute_cost(A2, Y, parameters)

            dA2 = -(np.divide(Y, A2) - np.divide(1-Y,1-A2))

            dA1, dW2, db2 = self._linear_activation_backward(dA2, cache2, activation="sigmoid")
            dA0, dW1, db1 = self._linear_activation_backward(dA1, cache1, activation="relu")

            if(self.regularization == "L2"):
                dW1 = dW1 + (self.lambd / m) * W1
                dW2 = dW2 + (self.lambd / m) * W2

            gradients["dW1"] = dW1
            gradients["db1"] = db1
            gradients["dW2"] = dW2
            gradients["db2"] = db2

            parameters = self._update_parameters(parameters, gradients)

            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]

            if(self.verbose and i%100==0):
                print "cost after iteration %d is : %f"%(i,cost)
                costs.append(cost)

        self.parameters_ = parameters
        self.costs_ = costs

    def _L_layer_model(self, X, Y):
        """
        Run the Deep Neural Network.
        """
        np.random.seed(2)
        costs = []
        m = X.shape[1]
        parameters = self._initialize_parameters_deep()
        for i in range(self.num_iterations):
            AL, caches = self._L_model_forward(X, parameters)
            cost = self._compute_cost(AL, Y, parameters)
            gradients = self._L_model_backward(AL, Y, caches, parameters)
            parameters = self._update_parameters(parameters, gradients)
            if(i%100 == 0 and self.verbose):
                print "Cost after iteration %d is : %f"%(i, cost)
                costs.append(cost)

        self.parameters_ =  parameters
        self.costs_ = costs

    def _predict(self, X):
        """
        Predict values using the test set.
        """
        try:
            self.parameters_
        except AttributeError:
            raise ValueError('fit(X, Y) needs to be called before using predict(X).')

        L = len(self.parameters_) // 2
        if(L > 2):
            AL, caches = self._L_model_forward(X, self.parameters_)
            predictions = np.round(AL)
        else:
            A1, cache1 = self._linear_activation_forward(X, self.parameters_["W1"], self.parameters_["b1"], activation="relu")
            A2, cache2 = self._linear_activation_forward(A1, self.parameters_["W2"], self.parameters_["b2"], activation="sigmoid")
            predictions = np.round(A2)
        return predictions

    def _accuracy(self, predictions, Y):
        """
        Calculates the accuracy.
        """
        accuracy = (100 - np.mean(np.abs(predictions-Y))*100)
        return accuracy

    def _evaluate_loss(self):
        """
        Stores the costs.
        """
        try:
            self.parameters_
        except:
            raise ValueError('fit(X, Y) needs to be called before using evaluate_loss().')
        return self.costs_
