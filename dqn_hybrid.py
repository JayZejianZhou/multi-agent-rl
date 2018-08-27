import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import to5

class DQN:
    def __init__(self, n_actions, state_size, gamma=0.9, learning_rate=0.001,
                 eps_greedy=0.5, eps_increment=1e-5, replace_target_freq=2000):
        self.n_actions = n_actions
        self.state_size = state_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eps_greedy = eps_greedy
        self.eps_increment = eps_increment
        self.learning_step = 0
        self.replace_target_freq = replace_target_freq
        self.eval_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_weights()

    def build_network(self):
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))

        return model

    def update_target_weights(self):
        self.target_network.set_weights(self.eval_network.get_weights())

    def choose_action(self, state):
        p = np.random.random()
        if p < self.eps_greedy:
            action_probs = self.eval_network.predict(state[np.newaxis, :])
            return np.argmax(action_probs[0])
        else:
            return random.randrange(self.n_actions)

    def learn(self, states, actions, rewards, states_next, done,cooperative=True):
        if self.learning_step % self.replace_target_freq == 0:
            self.update_target_weights()

        rows = np.arange(done.shape[0])
        not_done = np.logical_not(done)

        eval_next = self.eval_network.predict(states_next)
        target_next = self.target_network.predict(states_next)
        discounted_rewards=[]
        if cooperative:
            #JAl for cooperative
            discounted_rewards = self.gamma * \
                target_next[rows, np.argmax(eval_next, axis=1)]
        else:
            #minimax for noncooperative
            for state_ev in eval_next:
                q_value_split=np.zeros((5,5))
                for i in range(5):
                    for j in range(5):
                        q_value_split[i][j]=state_ev[to5.to_five(i,j)]
                #player 2 is noncooperative, now minimax him
                minned=[]
                for i in range(5):
                    minned.append(np.min(q_value_split[i]))
                discounted_rewards.append(self.gamma*max(minned))
            discounted_rewards=np.asarray(discounted_rewards)
            
            
        y = self.eval_network.predict(states)
        y[rows, actions] = rewards
        y[not_done, actions[not_done]] += discounted_rewards[not_done]

        history = self.eval_network.fit(states, y, epochs=1, verbose=0)
        self.learning_step += 1
        if self.eps_greedy < 0.9:
            self.eps_greedy += self.eps_increment

        return history

    def load(self, name):
        self.eval_network.load_weights(name)
        self.update_target_weights()

    def save(self, name):
        self.eval_network.save_weights(name)
