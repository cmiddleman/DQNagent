
from collections import deque
import numpy as np


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import dropout


class Memory:

    def __init__(self,end_state_repeat = 4, capacity=4096):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.end_state_repeat = end_state_repeat

    def add(self, state, action, reward, next_state, is_end):
        self.memory.append((state, action, reward, next_state, is_end))
        if is_end:
            for _ in range(self.end_state_repeat):
                self.memory.append((state, action, reward, next_state, is_end))

    def sample_experiences(self, batch_size):
        assert batch_size <= len(self), 'batch_size larger than memory'
        #return np.random.choice(self.memory, batch_size)
        indices = np.random.randint(len(self), size=batch_size)
        batch = [self.memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)
        ]
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
      return len(self.memory)


class Network:

    def __init__(self, eta, tau, dropout=False):
        self.eta = eta
        self.tau = tau

        self.model = self.make_model()
        self.target_model = self.make_model()
        self.target_model.set_weights(self.model.get_weights())

    def make_model(self, ONE_HOT_STATE_SIZE=3, BOARD_SIZE=9):
        model = Sequential()
        model.add(InputLayer(input_shape=(BOARD_SIZE,ONE_HOT_STATE_SIZE)))
        model.add(Flatten())
        model.add(Dense(256, activation='elu'))
        if dropout:
            model.add(Dropout(.2))
        model.add(Dense(256, activation='elu'))
        if dropout:
            model.add(Dropout(.2))
        model.add(Dense(256, activation='elu'))
        model.add(Dense(BOARD_SIZE, activation='linear'))

        model.compile(loss='mean_squared_error',
                      optimizer=Adam(learning_rate=self.eta))

        return model

        

    def update_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

