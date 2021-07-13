
from collections import deque
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer

class Memory:

    def __init__(self,end_state_repeat = 4, capacity=4096):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.end_state_repeat = end_state_repeat

    def add(self, state, action, reward, nextState, isEnd):
        self.memory.append((state, action, reward, nextState, isEnd))
        if isEnd:
            for _ in range(self.end_state_repeat):
                self.memory.append((state, action, reward, nextState, isEnd))

    def getBatch(self, batchSize):
        assert batchSize <= len(self.memory), 'batchsize larger than memory'
        return np.random.choice(self.memory, batchSize)
    
    def __len__(self):
      return len(self.memory)


class Network:

    def __init__(self, eta=.005, tau=1/16):
        self.eta = eta
        self.tau = tau

        self.model = self.makeModel()
        self.tarModel = self.makeModel()

    def makeModel(self, ONE_HOT_STATE_SIZE=3, BOARD_SIZE=9):
        model = Sequential()
        model.add(InputLayer(input_shape=(ONE_HOT_STATE_SIZE,BOARD_SIZE)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(BOARD_SIZE, activation='linear'))

        model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.Adam(learning_rate=self.eta))

        return model

    def updateTarModel(self):
        weights = self.model.get_weights()
        tarWeights = self.tarModel.get_weights()
        for i in range(len(weights)):
            tarWeights[i] = weights[i] * self.tau + tarWeights[i] * (1 - self.tau)
        self.tarModel.set_weights(tarWeights)