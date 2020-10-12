from lib.dqn import DQNAgent
from lib.utils import *
import random

# === GLOBAL CONSTANT
EPISODES = 10
TIME = 1500
TRAINING_THR = 0.01

# === SETTING ENVIRONMENT
actions = [0, 1, 2, 3, 4]
dataset = fakeDataset(Nsamples=TIME)
state_size = dataset.shape[1]
action_size = len(actions)

agent = DQNAgent(state_size, action_size)
train, test = agent.data.train_test_split(dataset)

y_train = train[:, 4]
train_sc, scalar = standardScalar(train)
train_sc[:, 4] = y_train
print(f"train shape: {train.shape}")
done = False
batch_size = 32
stopCondition = False
initStep = 1

for e in range(EPISODES):
    state = train_sc[0]
    state = np.reshape(state, [1, agent.state_size])
    agent.total_rewards = 0
    agent.epsilon = 1 * random.choice([0, 1]) * agent.epsilon_decay
    if stopCondition:
        """Loss < TrainingThr, so try to learn unseen samples."""
        initStep = int(TIME * 0.8 * 0.8) + 1
        state = train_sc[initStep]
        state = np.reshape(state, [1, agent.state_size])
    for time in range(TIME):
        score = (agent.total_rewards / (time + 1)) * 100
        agent.data.measures['score'][e] = score
        print(f'*** TIME: {time} *** score rewards %: {(score)}')
        action = agent.act(state)  # index of maximum the maximum Q-learning value
        print(f"Given behaviour: {state[0, agent.state_size - 1]}, chosen action: {action}")
        reward, dist = agent.step(action, state)
        agent.data.measures['totalRewards'][e].append(agent.total_rewards)
        print(f"Get reward: {reward}, given dist: {dist}")
        done = True if time + initStep == int(train.shape[0] * 0.8) else False
        if not done:
            next_state = np.reshape(train_sc[initStep + time], [1, agent.state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
        else:
            agent.update_target_model()
            print("=== Sharing weights between models. ===")
            print("episode: {}/{}, score: {}, epsilon: {:.2}"
                  .format(e, EPISODES, score, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size=batch_size, episod=e)

        nMiniBatches = len(agent.data.measures['loss'][e])
        if nMiniBatches >= batch_size * 20 and nMiniBatches % batch_size == 0:
            print(f"Print Metrics miniBatch {nMiniBatches}")
            agent.plotMetrics(episod=e, nBathc=nMiniBatches)
            agent.plotLoss(episod=e)
            agent.plotRewards(episod=e)

        if len(agent.data.measures['loss'][e]) > 0 and agent.data.measures['loss'][e][-1] <= TRAINING_THR:
            print(f"Minimum Loss: {agent.data.measures['loss'][e]}")
            stopCondition = True
            break

        # if e == 5:
        #    """try to learn new samples"""
        #    stopCondition = True

print("=== Hyperparameters ===")
print(f"LR: {agent.learning_rate}; Gamma: {agent.gamma}, Eps: {agent.epsilon}, clip: {agent.clipDelta}")
print(f"Final Scores (totRewards/steps) for episodes: {agent.data.measures['score']}")