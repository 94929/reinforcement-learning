import gym
import random
import numpy as np
import matplotlib.pyplot as plt

def true_action(p, action):
    action_space = [0, 1, 2, 3]
    if random.random() < p:
        return action
    else:
        action_space.remove(action)
        return random.choice(action_space)

def step(state, action):
    if state not in [0, 1, 3, 4, 5, 6, 7, 8, 9]:
        print('current state: ', state)
        raise ValueError('state?')
    if action not in [0, 1, 2, 3]:
        raise ValueError('action?')

    real_action = true_action(0.5, action)

    ns = -1
    if state == 0:
        poss_nss = [0, 1, 4]
        if real_action == 0 or real_action == 3:
            ns = 0
        elif real_action == 1:
            ns = 4
        elif real_action == 2:
            ns = 1
    elif state == 1:
        poss_nss = [0, 1, 2, 5]
        if real_action == 0:
            ns = 0
        elif real_action == 1:
            ns = 5
        elif real_action == 2:
            ns = 2
        elif real_action == 3:
            ns = 1
    elif state == 3:
        poss_nss = [2, 3, 6]
        if real_action == 0:
            ns = 2
        elif real_action == 1:
            ns = 6
        elif real_action == 2 or real_action == 3:
            ns = 3
    elif state == 4:
        poss_nss = [0, 4, 5]
        if real_action == 0 or real_action == 1:
            ns = 4
        elif real_action == 2:
            ns = 5
        elif real_action == 3:
            ns = 0
    elif state == 5:
        poss_nss = [1, 4, 5, 7]
        if real_action == 0:
            ns = 4
        elif real_action == 1:
            ns = 7
        elif real_action == 2:
            ns = 5
        elif real_action == 3:
            ns = 1
    elif state == 6:
        poss_nss = [3, 6, 9]
        if real_action == 0 or real_action == 2:
            ns = 6
        elif real_action == 1:
            ns = 9
        elif real_action == 3:
            ns = 3
    elif state == 7:
        poss_nss = [5, 7, 8]
        if real_action == 0 or real_action == 1:
            ns = 7
        elif real_action == 2:
            ns = 8
        elif real_action == 3:
            ns = 5
    elif state == 8:
        poss_nss = [7, 8, 9, 10]
        if real_action == 0:
            ns = 7
        elif real_action == 1:
            ns = 10
        elif real_action == 2:
            ns = 9
        elif real_action == 3:
            ns = 8
    elif state == 9:
        poss_nss = [6, 8, 9]
        if real_action == 0:
            ns = 8
        elif real_action == 1 or real_action == 2:
            ns = 9
        elif real_action == 3:
            ns = 6
    
    # calculate the rest
    if ns == 2:
        reward = 1000
        done = True
    elif ns == 10:
        reward = -1000
        done = True
    else:
        reward = -1
        done = False

    return ns, reward, done, None

# Define our world to be non-deterministic(i.e. stochastic), 'is_slippery=True'
#env = gym.make('FrozenLake-v0')

# Initialise Q-Table with zeros representing rewards
nb_states = 11 #env.observation_space.n     # 11
nb_actions = 4 #env.action_space.n         # 4
Q = np.zeros([nb_states, nb_actions])

# Set learning rate, alpha representing how much the agent listen to Q-Table
alpha = .85

# Set discount reward factor, gamma
gamma = .99

# Set number of iterations
nb_episodes = 5500

# Create a list to contain cumulative rewards per episode
cumulative_rewards = []

for i in range(nb_episodes):
    print(i)
    
    state = 0 # Init env and receive initial state of the agent
    done = False        # Set done to False, is required to enter the loop

    # The Q-Table learning algorithm with discounted reward
    while not done:
        # Choose an action which maximises the reward (from the current state)
        # When choosing an action, use decaying random noise for path exploration
        # @type(action) == class numpy.int64 
        action = np.argmax(Q[state, :] + np.random.randn(1, nb_actions) / (i + 1))

        # Receive a feedback after taking the action
        # @int, float, bool, _
        #new_state, reward, done, _ = env.step(action)
        new_state, reward, done, _ = step(state, action)

        # In stochastic world, we need not to rely on previous Q-Table
        # This can be done by adding the learning rate variable, alpha
        # Q(s, a) = (1-alpha)*Q(s, a) + alpha*(reward + gamma*max(a').Q(s', a')
        Q[state, action] = (1 - alpha) * Q[state, action] \
                            + alpha * (reward + gamma * np.max(Q[new_state, :]))

        state = new_state
    
print(Q)

