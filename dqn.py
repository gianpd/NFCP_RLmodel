import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K


import tensorflow as tf

EPISODES = 6
TIME = 15000

BEST = np.array([1000, 10000, 500, 12, 1]).reshape(1, 5)
WORSTE = np.array([0, 0, 1, 50, 0]).reshape(1, 5)

plt.close('all')

class Measurement():

    def __init__(self, state_size, action_size, train_size, test_size):

        self._stateSize = state_size
        self._actionSize = action_size
        self._trainSize = train_size
        self._testSize = test_size
        self.measures = {'loss': [], 'accuracy': [], 'totalRewards': []}

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
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.89
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.total_rewards = 0
        self.data = Measurement(state_size, action_size, train_size=0.8, test_size=0.2)



    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        """Huber loss for Q Learning: it acts as a square loss for small errors and as a mean absolute loss for big errors

           References: https://en.wikipedia.org/wiki/Huber_loss
                       https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
           """
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        """Neural Network model for learning the Q(s,a) function. The model will learn how to take action given a state.
        The Q-learning function is a non linear function of type: Q:S x A -> R."""

        model = Sequential()
        model.add(Dense(14, input_dim=self.state_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(24, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(10, activation='relu'))
        #model.add(Dropout(rate=0.3))
        model.add(Dense(self.action_size, activation='linear'))  # Regression problem.
        model.compile(loss=self._huber_loss,
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
            return random.randrange(self.action_size)
        print("predicted choice")
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns index action

    def replay(self, batch_size):
        """"replay buffer technique: when a minibatch is available in the buffer the model is trained """

        minibatch = random.sample(self.memory, batch_size)
        print("Replay buffer memory: ")
        loss = []
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            #print(f"first state's behavior: {state[:, 4]}")
            #print(f"predicted target: {target}, action: {action}, reward: {reward}")
            #print(f"sum_target: {np.sum(target)}")
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                future_target = self.target_model.predict(next_state)[0]  # predict future Q-learning value
                #print(f"max futureTarget: {np.amax(future_target)}")
                target[0][action] = reward + self.gamma * np.amax(future_target)
                #print(f"target[action]: {target[0][action]}")
                #print(f"state: {state}, target: {target}")
            history = self.model.fit(state, target, epochs=1, verbose=0)
            loss.append(history.history['loss'])
        self.data.measures['loss'].append(np.linalg.norm(loss))
        print(f"Loss: {np.linalg.norm(self.data.measures['loss'])}")

        if self.epsilon >= self.epsilon_min:
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
            dist = np.linalg.norm(WORSTE - state)
            if dist > 5:  # not too bad
                if action == 3:
                    self.total_rewards += 1
                    return 1, dist
                else:
                    return 0, dist
            else:  # very bad
                if action == 4:
                    self.total_rewards += 1
                    return 1, dist
                else:
                    return 0, dist

        if state[0, 4] == 1:  # good node
            dist = np.linalg.norm(BEST - state)
            if dist > 5:  # not too good
                if action >= 1 and action < 3:
                    self.total_rewards += 1
                    return 1, dist
                else:
                    return 0, dist
            else:  # very good
                if action == 1:
                    self.total_rewards += 1
                    return 1, dist
                else:
                    return 0, dist

    def plotLoss(self, episod=0):

        epochs_loss = range(len(self.data.measures['loss']))
        plt.plot(epochs_loss, self.data.measures['loss'])
        plt.grid()
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.savefig(f'plots/Loss_{episod+len(self.data.measures["loss"])}.png')
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


def fakeDataset(Nsamples=1000):
    """Simulate a dataset containing states about nodes:
    Features = [UpTime, BalanceInTime, OutLinkMatrix, Ping, Behaviour]"""

    dataset = np.zeros((Nsamples, 5))
    for i in range(Nsamples):
        b = BEST + np.random.randn(5)*np.random.randint(1, 5)
        b[0, 4] = 1
        w = WORSTE + np.random.randn(5)*np.random.randint(1, 5)
        w[0, 4] = 0
        dataset[i, :] = b if np.random.choice([0,1]) == 0 else w

    return dataset

if __name__ == "__main__":

    actions = [0, 1, 2, 3, 4]
    dataset = fakeDataset(Nsamples=15000)
    state_size = dataset.shape[1]
    action_size = len(actions)

    agent = DQNAgent(state_size, action_size)
    train, test = agent.data.train_test_split(dataset)
    print(f"train shape: {train.shape}")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = train[0]
        state = np.reshape(state, [1, 5])
        agent.total_rewards = 0
        agent.epsilon = 1
        for time in range(TIME):
            print(f'*** TIME: {time} *** total rewards: {agent.total_rewards}')
            action = agent.act(state)  #index of maximum the maximum Q-learning value
            print(f"Given behaviour: {state[0, 4]}, chosen action: {action}")
            reward, dist = agent.step(action, state)
            print(f"Get reward: {reward}, given dist: {dist}")
            done = True if time+1 == int(train.shape[0]) else False
            if not done:
                next_state = np.reshape(train[time+1], [1, 5])
                agent.memorize(state, action, reward, next_state, done)
                state = next_state
            else:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, epsilon: {:.2}"
                      .format(e, EPISODES, (agent.total_rewards/time)*100, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                if len(agent.data.measures['loss']) % 150 == 0:
                    agent.plotLoss(episod=e)
                    #agent.data.measures['loss'].clear()
            agent.data.measures['totalRewards'].append(agent.total_rewards)

        agent.plotRewards(episod=e)