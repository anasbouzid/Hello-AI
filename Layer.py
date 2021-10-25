import numpy as np


class Layer:

    def __init__(self, neurons, activation, weights=[], biases=[]):
        self.neurons = neurons
        self.activation = activation
        self.weights = np.array(weights) if weights else None
        self.biases = np.array(biases) if biases else None
        self.inputs = None
        # self.out = None

        self.delta_weights = None
        self.delta_biases = None
        self.total_delta_weights_added = 0
        self.total_delta_biases_added = 0

    def set_weights(self, weights):
        self.weights = np.array(weights)

    def set_biases(self, biases):
        self.biases = np.array(biases)

    def forward_propagate(self, inp):
        # out = σ(zW + b)
        if(self.activation.name == "identity"):
            self.inputs = np.array(inp)
        else:
            self.inputs = np.array(inp) @ self.weights.T + self.biases

        return self.out

    def update_weights_and_biases(self, optimizer, l2_regularization):
        if (l2_regularization > 0):
            self.weights = self.weights * (1 - l2_regularization)

        average_delta_weights = self.delta_weights / self.total_delta_weights_added
        average_delta_biases = self.delta_biases / self.total_delta_biases_added

        self.weights = optimizer.optimize(self.weights, average_delta_weights)
        self.biases = optimizer.optimize(self.biases, average_delta_biases)
        # self.weights = np.round(self.weights, 8)

        self.delta_weights= None
        self.delta_biases= None
        self.total_delta_weights_added = 0
        self.total_delta_biases_added = 0


    def add_delta_weights(self, delta_weights):
        if(self.delta_weights is None):
            self.delta_weights = delta_weights
        else:
            self.delta_weights = self.delta_weights + delta_weights

        self.total_delta_weights_added += 1

    def add_delta_biases(self, delta_biases):
        if(self.delta_biases is None):
            self.delta_biases = delta_biases
        else:
            self.delta_biases = self.delta_biases + delta_biases

        self.total_delta_biases_added += 1

    @property
    def out(self):
        return self.activation.fn(self.inputs)
