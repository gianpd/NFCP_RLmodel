import numpy as np
from sklearn.preprocessing import StandardScaler

# ==== GLOBAL VARIABLES ====
BEST = np.array([1000, 10000, 500, 12, 1]).reshape(1, 5)
WORSTE = np.array([0, 0, 1, 5, 0]).reshape(1, 5)


def fakeDataset(Nsamples=1000):
    """Simulate a dataset containing states about nodes:
    Features = [UpTime, BalanceInTime, OutLinkMatrix, Ping, Behaviour]"""

    dataset = np.zeros((Nsamples, 5))
    for i in range(Nsamples):
        b = BEST - np.random.randn(5)*np.random.randint(1, 5)
        b[0, 4] = 1
        w = WORSTE + np.random.randn(5)*np.random.randint(1, 5)
        w[0, 4] = 0
        dataset[i, :] = b if np.random.choice([0,1]) == 0 else w

    return dataset

def standardScalar(X):
    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)
    return X_scaled, scalar

def computeLabels(states, worste, best):

    labels = []
    for state in states:
        if state[0, 4] == 0:
            dist = np.linalg.norm(worste - state)
            if dist > 2.4:
                labels.append(3.0)
            else:
                labels.append(4.0)
        if state[0, 4] == 1:  # good node
            dist = np.linalg.norm(best - state)
            if 2 < dist < 2.4:  # not too good
                labels.append(1.0)
            if dist >= 2.4:
                labels.append(2.0)
            if dist <= 2.0:  # very good
                labels.append(0.0)

    return labels