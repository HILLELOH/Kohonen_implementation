# kohonen

import numpy as np
import matplotlib.pyplot as plt

class Kohonen:
    def __init__(self, map_shape, learning_rate=0.1, sigma=1.0):
        self.map_shape = map_shape
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.weights = np.random.rand(map_shape[0], map_shape[1], 2) # Initialize the weight matrix

    def find_best_matching_unit(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=2) # Euclidean distance between input_vector and each weight vector
        bmu_index = np.unravel_index(np.argmin(distances), distances.shape) # Find the index of the best matching unit (BMU)
        return bmu_index

    def update_weights(self, input_vector, bmu_index):
        distance_squares = np.sum((np.indices(self.map_shape) - np.array(bmu_index)[:, np.newaxis, np.newaxis]) ** 2, axis=0)
        neighborhood = np.exp(-distance_squares / (2 * self.sigma ** 2))
        self.weights += self.learning_rate * neighborhood[:, :, np.newaxis] * (input_vector - self.weights)

    def train(self, input_data, num_epochs):
        for epoch in range(num_epochs):
            np.random.shuffle(input_data)

            for input_vector in input_data:
                bmu_index = self.find_best_matching_unit(input_vector)
                self.update_weights(input_vector, bmu_index)

    def get_weights(self):
        return self.weights
    

def cube(distribution='uniform'):
    """{(x, y) | x, y ∈ [0, 1]})}"""
    global input_data
    epochs = [1, 50]
    
    if distribution == 'uniform':
        input_data = np.random.uniform(size=(3000, 2))

    elif distribution == 'normal':
        input_data = np.random.normal(loc=0.5, scale=0.2, size=(3000, 2))
        input_data = np.clip(input_data, 0, 1)

    elif distribution == 'geometric':
        p = 0.5
        input_data = np.random.geometric(p, size=(3000, 2))
        input_data = input_data / np.max(input_data)  # Scale input_data to the range (0, 1)

    for epoch in epochs:
        kohonen_few_neurons = Kohonen(map_shape=(4, 5), learning_rate=0.1, sigma=1.0)
        kohonen_few_neurons.train(input_data, num_epochs=epoch)
        weights_few_neurons = kohonen_few_neurons.get_weights()

        kohonen_many_neurons = Kohonen(map_shape=(20, 10), learning_rate=0.1, sigma=1.0)
        kohonen_many_neurons.train(input_data, num_epochs=epoch)
        weights_many_neurons = kohonen_many_neurons.get_weights()


        plt.scatter(input_data[:, 0], input_data[:, 1], color='blue', label='Input Data')
        plt.scatter(weights_few_neurons[:, :, 0], weights_few_neurons[:, :, 1], color='red', label='Learned Weights (Few Neurons)')
        plt.title(f'Kohonen Network - Few Neurons, epochs: {epoch}')
        plt.legend(loc='upper left')
        plt.show()


        plt.scatter(input_data[:, 0], input_data[:, 1], color='blue', label='Input Data')
        plt.scatter(weights_many_neurons[:, :, 0], weights_many_neurons[:, :, 1], color='red', label='Learned Weights (Many Neurons)')
        plt.title(f'Kohonen Network - Many Neurons, epochs: {epoch}')
        plt.legend(loc='upper left')
        plt.show()

def daunat(distribution='uniform'):
    """{(x, y) | 4<=x^2 + y^2 <= 16}"""

    global r1, theta
    epochs = [1, 500]
    num_points = 3000

    if distribution == 'uniform':
        r1 = np.sqrt(np.random.uniform(size=num_points) * 12 + 4)
        theta = np.random.uniform(low=0, high=2 * np.pi, size=num_points)

    elif distribution == 'normal':
        pass

    elif distribution == 'geometric':
        pass

    for epoch in epochs:
        x = r1 * np.cos(theta)
        y = r1 * np.sin(theta)

        input_data = np.column_stack((x, y))
        kohonen_many_neurons = Kohonen(map_shape=(50, 6), learning_rate=0.1, sigma=1.0)
        kohonen_many_neurons.train(input_data, num_epochs=epoch)
        weights_many_neurons = kohonen_many_neurons.get_weights()

        plt.scatter(input_data[:, 0], input_data[:, 1], color='blue', label='Input Data')
        plt.scatter(weights_many_neurons[:, :, 0], weights_many_neurons[:, :, 1], color='red', label='Learned Weights')
        plt.title(f'Kohonen Network - 300 Neurons - epochs: {epoch}')
        plt.legend(loc='upper left')
        plt.show()

if __name__ == '__main__':
    cube(distribution='uniform')

