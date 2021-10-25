from Layer import Layer
import numpy as np
from CostFunction import CostFunction
import pickle

class NeuralNetwork:

    def __init__(self, layers, optimizer, costFunction=CostFunction.quadratic, l2_regularization=0):
        self.layers = layers
        self.optimizer = optimizer
        self.costFunction = costFunction
        self.l2_regularization = l2_regularization

    def total_neurons(self):
        return sum(layer.neurons for layer in self.layers)

    def weights(self):
        return np.array([layer.weights for layer in self.layers]).flat

    def forward_propagate(self, inp):
        result = inp
        for layer in self.layers:
            result = layer.forward_propagate(result)

        return result

    def cost(self, actual, expected):
        # return self.costFunction.fn(actual, np.array(expected))
        return np.sum(self.costFunction.fn(actual, np.array(expected)))


    def learn(self, inp, expected):
        actual = self.forward_propagate(inp)

        self.back_propogation(expected)
        cost = self.cost(actual, expected)

        return (actual, cost)

    def back_propogation(self, expected):

        # ∂C/∂O is a direvative of the cost function
        dCdO = self.costFunction.dFn(self.layers[-1].out, expected)

        # iterate backwards
        for i, layer in enumerate(reversed(self.layers)):
            if(not self.__has_previous_layer(i)):
                break

            # ∂C/∂O 0.74136507
            # ∂O/∂I 0.21778834 -> 0.186815602
            # ∂I/∂W 0.59326999

            # ∂C/∂W = ∂I/∂W * ∂O/∂I * ∂C/∂O
            #       = previous_output * derivate_activation_function * derivate_cost_function
            dIdW = self.__get_previous_layer(i).out
            dOdI = layer.activation.dFn(layer.out)

            # dCdO.elementProduct(dFn(out))

            # double x = out.elementProduct(dCdO).sumElements()
            # Vec sub = dCdO.sub(x)
            # return out.elementProduct(sub)

            if (layer.activation.name != "softmax"):
                dCdI = dCdO * dOdI
            else:
                # sub = dCdO - np.sum(layer.out * dCdO)
                # dCdI = layer.out * sub
                dCdI = np.sum(dCdO * dOdI, axis=1)

            dCdW = np.outer(dCdI, dIdW)

            layer.add_delta_weights(dCdW)
            layer.add_delta_biases(dCdI)

            # ∂C/∂O_h = Σ(∂C(lastLayer_i)/∂O_h) # where ∂O_h == ∂O_hiddenlayer
            #         = Σ(∂I/∂O * ∂C/dI)
            #         = Σ(weight * ∂C/dI) # where ∂C/dI already calculted
            dCdO = layer.weights.T @ dCdI

    def update_from_learning(self):
        for layer in self.layers[1:]:  # skip the first layer (input layer)
            layer.update_weights_and_biases(self.optimizer, self.l2_regularization)

    def init_random_weights(self):
        for i, layer in enumerate(reversed(self.layers)):
            if(not self.__has_previous_layer(i)):
                break
            shape = (layer.neurons, self.__get_previous_layer(i).neurons)
            weights = np.random.uniform(-0.5, 0.5, size=shape)
            biases = np.zeros(layer.neurons)
            layer.set_weights(weights)
            layer.set_biases(biases)


    def __get_previous_layer(self, index):
        return self.layers[-index - 2]

    def __has_previous_layer(self, index):
        return len(self.layers) > index + 1

    def save_to_file(self, name="NeuralNetwork", path="mnist-example/trained_models"):
        obj = {
            'biases': [layer.biases for layer in self.layers],
            'weights': [layer.weights for layer in self.layers],
        }
        with open(f"{path}/{name}.pkl", "wb") as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_from_file(self, name="NeuralNetwork"):
        with open(f"{name}.pkl", "rb") as f:
            data = pickle.load(f)
            for i, layer in enumerate(self.layers):
                layer.set_weights(data["weights"][i])
                layer.set_biases(data["biases"][i])
