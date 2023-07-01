import numpy as np
import matplotlib.pyplot as plt
from my_neural_network import NeuralNetwork


def scale_input(input: list) -> np.ndarray:
    '''scale the input to be between 0.01 and 1.00
    Dividing the raw inputs by 255.0 will scale them down to the range 0.01 to 0.99.
    We need to multiply them by 0.99 to bring them into the desired range of 0.01 to 0.99.
    Finally, we need to add 0.01 to shift them up to the desired range of 0.01 to 1.00.

    '''
    return (np.asfarray(input) / 255.0 * 0.99) + 0.01


if __name__ == '__main__':

    # Training with the full datasets
    train_data_path = 'wkd/inputs/mnist_train.csv'
    test_data_path = 'wkd/inputs/mnist_test.csv'

    training_data_file = open(train_data_path, 'r')
    training_data_list = training_data_file.readlines()[:1000]
    training_data_file.close()
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    # learning rate
    learning_rate = 0.1
    # create instance of neural network

    neural_network = NeuralNetwork(
        input_nodes=input_nodes, hidden_nodes=hidden_nodes,
        learning_rate=learning_rate, output_nodes=output_nodes)

    for row in training_data_list:
        all_values = row.split(',')
        inputs = scale_input(all_values[1:])
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        neural_network.train(inputs, targets)

    teste_data_file = open(test_data_path, 'r')
    test_data_list = teste_data_file.readlines()[:100]
    teste_data_file.close()
    scorecard = []
    for row in test_data_list:
        all_values = row.split(',')
        correct_label = int(all_values[0])
        print(correct_label, 'correct label')
        inputs = scale_input(all_values[1:])
        outputs = neural_network.query(inputs)
        label = np.argmax(outputs)
        print(label, 'network\'s answer:')
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    print(f'Accuracy: {sum(scorecard)/len(scorecard)*100}%')
