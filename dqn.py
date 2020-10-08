import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K


import tensorflow as tf

EPISODES = 3

class DQNAgent:
    """Neural Fairness Consensus Protocol Deep-Q-Network model:
    nodes (agents) have to select an action for each state. States are samples reporting some features about
    the behaviour of each node. Actions indicate how much a node has the right to participate to the final consensus.
    The agent can choose between 5 actions:
    1) participate with 100% tickets: 0;
    2) participate with 75% tickets: 1;
    3) participate with 50% tickets: 2;
    4) participate with 25% tickets: 3;
    5) participate with 0% tickets: 4"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.7  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.89
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.total_rewards = 0



    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        """Huber loss for Q Learning

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
        The model's inputs are states, while the outputs are actions. In particular the outputs are 5 softmax function
        outputs. The selected action will be the action with the biggest associated probability."""
        model = Sequential()
        model.add(Dense(14, input_dim=self.state_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(24, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.3))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """epsilon-greedy method"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        """"replay buffer technique"""

        minibatch = random.sample(self.memory, batch_size)
        k = 0
        for state, action, reward, next_state, done in minibatch:
            k += 1
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            history = self.model.fit(state, target, epochs=1, verbose=0)
            if k % 300 == 0:
                loss = history.history['loss']
                epochs = range(1, len(loss) + 1)
                plt.plot(epochs, loss)
                plt.title('Training loss')
                plt.savefig(f'plots/loss_{k}.png')
                plt.close()


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def step(self, action, state):
        reward = self.reward(state, action)
        return reward

    def reward(self, state, action):

        if state[0, 4] == 0:
            if action >= 3:
                self.total_rewards += 1
                return 1
            else:
                return 0

        if state[0, 4] == 1:
            if action < 3:
                return 1
            else:
                return 0

def fakeDataset(Nsamples=1000):
    """Simulate a dataset containing states about nodes:
    Features = [UpTime, BalanceInTime, OutLinkMatrix, Ping, Behaviour]"""

    best = [1000, 10000, 500, 12, 1]
    worste = [0, 0, 1, 50, 0]

    dataset = np.zeros((Nsamples, 5))
    for i in range(Nsamples):
        b = best + np.random.randn(5)*np.random.randint(1, 5)
        b[4] = 1
        w = worste + np.random.randn(5)*np.random.randint(1, 5)
        w[4] = 0
        dataset[i, :] = b if np.random.choice([0,1]) == 0 else w

    return dataset






if __name__ == "__main__":

    actions = [0, 1, 2, 3, 4]

    dataset = fakeDataset()
    state_size = dataset.shape[1]
    action_size = len(actions)

    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = dataset[0]
        state = np.reshape(state, [1, 5])
        for time in range(500):
            action = agent.act(state)
            print(f"Given behaviour: {state[0, 4]}, chosen action: {action}")
            reward = agent.step(action, state)
            print(f"Get reward: {reward}")
            done = 1 if time+1 == int(dataset.shape[0]*0.5) else 0
            next_state = np.reshape(dataset[time+1], [1, 5])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)