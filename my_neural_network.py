import numpy as np


class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int, learning_rate: float) -> None:
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.w_input_hodden = np.random.normal(0, pow(self.hidden_nodes, -0.5))
        self.w_hidden_output = np.random.normal(
            0, pow(self.output_nodes, -0.5))

    def train():
        '''refine the weights after being given a training set example to learning from'''
        pass

    def query():
        '''give an answer from the output nodesafter being given an input'''
        pass


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
