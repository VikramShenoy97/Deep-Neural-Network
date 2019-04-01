import numpy as np
from load_data import load_dataset
from neural_network import NeuralNetwork
from draw_graph import drawGraph


train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.
n_x = train_x.shape[0]
layers_dims = [n_x, 20, 5, 7, 1]  # 4-Layer Neural Network
#layers_dims = [n_x, 7, 1]        # 2-Layer Neural Network
number_of_epochs = 2500
NN = NeuralNetwork(layers_dims, learning_rate=0.0075, num_iterations=number_of_epochs, verbose=True, regularization="L2")
NN.fit(train_x, train_y)
training_predictions = NN.predict(train_x)
training_loss = NN.evaluate_loss()
training_accuracy = NN.accuracy(training_predictions, train_y)
print "Training Accuracy = %f " %(training_accuracy) + "%"
testing_predictions = NN.predict(test_x)
testing_accuracy = NN.accuracy(testing_predictions, test_y)
print "Testing Accuracy = %f " %(testing_accuracy) + "%"
drawGraph(number_of_epochs, training_loss, training_accuracy, testing_accuracy)
