# NFCP_RLmodel
***A Deep-Q-Network model for blockchain environment.***
Agents learn to assign a behavior value to other agents looking at agents states containing agents features.

### Deep-Q-Learning
nodes (agents) have to select an action for each state. States are samples reporting some features about
the behaviour of each node. Actions indicate how much a node has the right to participate to the final consensus.
The agent can choose between 5 actions:
```
     participate with 100% tickets: 0;
     participate with 75% tickets: 1;
     participate with 50% tickets: 2;
     participate with 25% tickets: 3;
     participate with 0% tickets: 4
```
Collect states:
```
StateNodeA = [1000, 40, 580, ...]
StateNodeB = [1050, 10, 340, ...]
StateNodeC = [1000, 28, 100, ...]
StateNodeD = [1050, 10, 340, ...]
```
A Deep Neural Network is used to approximating the Q-learning function: 
<a href="https://www.codecogs.com/eqnedit.php?latex=Q:&space;S&space;\times&space;A&space;\rightarrow&space;\mathbb{R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q:&space;S&space;\times&space;A&space;\rightarrow&space;\mathbb{R}" title="Q: S \times A \rightarrow \mathbb{R}" /></a>



