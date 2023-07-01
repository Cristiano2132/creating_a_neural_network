import numpy as np
import scipy


class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int, learning_rate: float) -> None:
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.weight_input_hodden = np.random.normal(
            0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.weight_hidden_output = np.random.normal(
            0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        self.activation_function = self.__activation_function

    def __activation_function(self, x):
        return scipy.special.expit(x)

    def train(self, inputs_list: list, targets_list: list) -> None:
        '''refine the weights after being given a training set example to learning from'''
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.weight_input_hodden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.weight_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weight_hidden_output.T, output_errors)
        self.weight_hidden_output += self.learning_rate * \
            np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                   np.transpose(hidden_outputs))

    def query(self, inputs: list) -> np.ndarray:
        '''give an answer from the output nodes after being given an input'''
        inputs = np.array(inputs, ndmin=2).T
        hidden_inputs = np.dot(self.weight_input_hodden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.weight_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


if __name__ == "__main__":
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3

    nn_params = {
        'input_nodes': input_nodes,
        'hidden_nodes': hidden_nodes,
        'output_nodes': output_nodes,
        'learning_rate': learning_rate
    }

    neural_network = NeuralNetwork(**nn_params)

    print(neural_network.weight_input_hodden)
    print(neural_network.weight_hidden_output)
    inputs = np.array([1.0, 0.5, -1.5])
    print(neural_network.query(inputs))
