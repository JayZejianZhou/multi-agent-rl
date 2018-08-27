import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class NN:
    def __init__(self, input_size, output_size):
        self.output_size = output_size
        self.input_size = input_size
        self.network = self.build_network()
        self.update_target_weights()

    def build_network(self):
        model = Sequential()
        model.add(Dense(50, input_dim=self.input_size, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.output_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(0.001))

        return model

    def update_target_weights(self):
        self.network.set_weights(self.network.get_weights())

    def learn(self, inputs, ground_truth):          
        history = self.network.fit(inputs, ground_truth, epochs=100, verbose=0)
        return history

    def load(self, name):
        self.network.load_weights(name)
        self.update_target_weights()

    def save(self, name):
        self.network.save_weights(name)
