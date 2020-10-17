"""Test script"""
from keras.models import load_model
import pandas as pd

from lib.dqn import WORSTE_SC, BEST_SC
from lib.utils import computeLabels

try:
    model = load_model('models/SeLuDL2.h5')
    test = pd.read_csv('test/test_SeLuDL2.csv')
    labels = computeLabels(test, worste=WORSTE_SC, best=BEST_SC)
    # === EVALUATE MODEL
    score = model.evaluate(test, labels, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
except:
    print("Required model and/or test not available.")


