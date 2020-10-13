#!/usr/bin/python
import random
import numpy as np
from functools import partial
from collections import deque
import matplotlib.pyplot as plt

np.random.seed(36)
random.seed(36)

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

from lib.utils import *

print(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")

EPISODES = 7
TIME = 1000
TRAINING_THR = 0.01
BEST_SC = StandardScaler().fit_transform(BEST)
WORSTE_SC = StandardScaler().fit_transform(WORSTE)
plt.close('all')

class Measurement():

    def __init__(self, state_size, action_size, train_size, test_size):

        self._stateSize = state_size
        self._actionSize = action_size
        self._trainSize = train_size
        self._testSize = test_size
        self.measures = {'loss': {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}, 'accuracy': [],
                         'totalRewards': {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []},
                         'score': {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}}

    def train_test_split(self, dataset):

        train = dataset[0:int(dataset.shape[0]*self._trainSize)]
        test = dataset[int(dataset.shape[0]*self._trainSize):-1]
        return train, test


class DQNAgent:
    """Neural Fairness Consensus Protocol Deep-Q-Network model:
    nodes (agents) have to select an action for each state. States are samples reporting some features about
    the behaviour of each node. Actions indicate how much a node has the right to participate to the final consensus.
    The agent can choose between 5 actions:
    1) participate with 100% tickets: 0;
    2) participate with 75% tickets: 1;
    3) participate with 50% tickets: 2;
    4) participate with 25% tickets: 3;
    5) participate with 0% tickets: 4

    ref. Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).

    A DNN is used for approximating the Q-learning function. Two models are built: The first one for predicting the
    Q-learning at the current time (given the current action), the second one (target_model) for predicting the future
    Q-learning value (given the future state and action), in according to: Q(s,a) = r + gamma*Q'(s', a').
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.92
        self.learning_rate = 0.0001
        self.clipDelta = 1.0
        self.regDense = partial(tf.keras.layers.Dense,
                                activation="relu",
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                activity_regularizer=tf.keras.regularizers.l2(1e-4))
        self.model = self._build_model()
        #tf.keras.utils.plot_model(self.model, 'model.png', show_shapes=True)
        self.target_model = self._build_model()
        self.update_target_model()
        self.total_rewards = 0
        self.data = Measurement(state_size, action_size, train_size=0.8, test_size=0.2)



    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        """Huber loss for Q Learning: it acts as a square loss for small errors and as a mean absolute loss for big errors

           References: https://en.wikipedia.org/wiki/Huber_loss
                       https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
           """
        self.clipDelta = clip_delta
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        linear_loss = clip_delta * (K.abs(error) - clip_delta*0.5)
        return K.mean(tf.where(cond, squared_loss, linear_loss))

    def _build_model(self):
        """Neural Network model for learning the Q(s,a) function. The model will learn how to take action given a state.
        The Q-learning function is a non linear function of type: Q:S x A -> R."""

        model = Sequential()
        model.add(Dense(14, input_dim=self.state_size,
                        activation='relu',
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                        bias_regularizer=tf.keras.regularizers.l2(1e-4),
                        activity_regularizer=tf.keras.regularizers.l2(1e-4)
                        ))
        #model.add(Dense(14, input_dim=self.state_size, activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(self.regDense(24))
        #model.add(Dense(24, activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(self.regDense(16))
        #model.add(Dense(16, activation='relu'))
        #model.add(BatchNormalization())
        model.add(self.regDense(self.action_size, activation='linear'))
        model.add(Dense(self.action_size, activation='linear'))  # Regression problem.
        model.compile(loss=tf.keras.losses.Huber(
        delta=self.clipDelta, name='huber_loss'),
        optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model


    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        """Buffer memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """epsilon-greedy method"""
        if self.epsilon > np.random.rand(1)[0]:
            print("random choice")
            return np.random.randint(self.action_size)
            #return np.random.choice([1,2])
        print("predicted choice")
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns index action

    def replay(self, batch_size, episod=0):
        """"replay buffer technique: when a minibatch is available in the buffer the model is trained """

        minibatch = random.sample(self.memory, batch_size)
        print("Replay buffer: ")
        loss = []
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            print(f"first state's behavior: {state[:, 4]}")
            print(f"state: {state} next: {next_state}")
            print(f"predicted target: {target}")
            print(f"action: {action}, reward: {reward}")
            #print(f"sum_target: {np.sum(target)}")
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                future_target = self.target_model.predict(next_state)[0]  # predict future Q-learning value
                print(f"max futureTarget: {np.amax(future_target)}")
                target[0][action] = reward + self.gamma * np.amax(future_target)
                print(f"updated target: {target}, {target[0][action]}")
            history = self.model.fit(state, target, epochs=1, verbose=0)
            #print(len(history.history['loss']))
            loss.append(history.history['loss'])
        self.data.measures['loss'][episod].append(np.mean(loss))
        print(f"Loss: {self.data.measures['loss'][episod][-1]}")
        #nMiniBatches = len(self.data.measures['loss'][episod])
        #if nMiniBatches >= batch_size*20 and nMiniBatches % batch_size == 0:
        #    print(f"Print Metrics miniBatch {nMiniBatches}")
        #    self.plotMetrics(episod=episod, nBathc=nMiniBatches)
        #    self.plotLoss(episod=episod)
        #    self.plotRewards(episod=episod)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.epsilon = max(self.epsilon, self.epsilon_min)
        print(f"Epsilon: {self.epsilon}")

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def step(self, action, state):
        reward, dist = self.reward(state, action)
        return reward, dist

    def reward(self, state, action):


        if state[0, 4] == 0:  # bad node
            dist = np.linalg.norm(WORSTE_SC - state)
            if dist > 2.4:  # not too bad
                if action == 3:
                    self.total_rewards += 1
                    return 1, dist
                else:
                    #self.total_rewards -= 1
                    return -1, dist
            else:  # very bad
                if action == 4:
                    self.total_rewards += 1
                    return 1, dist
                else:
                    #self.total_rewards -= 1
                    return -1, dist

        if state[0, 4] == 1:  # good node
            dist = np.linalg.norm(BEST_SC - state)
            if 2 < dist < 2.4:  # not too good
                if action == 1:
                    self.total_rewards += 1
                    return 1, dist
                else:
                    return -1, dist
            if dist >= 2.4:
                if action == 2:
                    self.total_rewards += 1
                    return 1, dist
                else:
                    return -1, dist
            if dist <= 2.0:  # very good
                if action == 0:
                    self.total_rewards += 1
                    return 1, dist
                else:
                    #self.total_rewards -= 1
                    return -1, dist

    def plotLoss(self, episod=0):
        plt.close('all')
        batch = len(self.data.measures["loss"][episod])
        plt.close('all')
        epochs_loss = range(batch)
        plt.plot(epochs_loss, self.data.measures['loss'][episod])
        plt.grid()
        plt.title('Loss')
        plt.xlabel('training epochs')
        plt.savefig(f'plots/Loss_{episod+1}_{batch}.png')
        plt.close()

    def plotRewards(self, episod=0):
        plt.close('all')
        epochs_rewards = range(len(self.data.measures['totalRewards'][episod]))
        plt.plot(epochs_rewards, self.data.measures['totalRewards'][episod])
        plt.grid()
        plt.title(f'Episod: {episod+1}')
        plt.ylabel('Total Rewards')
        plt.xlabel('training epochs')
        plt.savefig(f'plots/TotRewards_{episod+1}.png')
        plt.close()

    def plotMetrics(self, episod=0, nBathc=0):

        plt.close('all')
        epochs_rewards = range(len(self.data.measures['totalRewards'][episod]))
        epochs_loss = range(len(self.data.measures["loss"][episod]))
        plt.plot(epochs_loss, self.data.measures['loss'][episod], label='loss')
        plt.plot(epochs_rewards, self.data.measures['totalRewards'][episod], label='TotRewards')
        plt.legend()
        plt.xlabel('training epochs')
        plt.title(f' Episode:{episod}; nBatch:{nBathc}; Lrate: {self.learning_rate}; '
                  f'score %: {self.data.measures["score"][episod]}')
        plt.grid()
        plt.savefig(f"plots/metrics_{episod}_{nBathc}.png")
        plt.close()

