import random
import numpy as np

def get_transition_matrices():
    # Each transition matrix is 11x11
    # Resulting tm is 4x(11x11)

    # Transition for Left (0)
    TL = np.array([
        [1,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
    ])

    # Transition for Down (1)
    TD = np.array([
        [0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,0],
    ])
    
    # Transition for Right (2)
    TR = np.array([
        [0,1,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,0],
    ])   
    
    # Transition for Up (3)
    TU = np.array([
        [1,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
    ])

    return np.dstack([TL, TD, TR, TU])

def true_action(p, action):
    action_space = [0, 1, 2, 3]
    if random.random() < p:
        return action
    else:
        action_space.remove(action)
        return random.choice(action_space)

def step(state, action, p):
    real_action = true_action(p, action)

    ns = -1
    if state == 0:
        if real_action == 0 or real_action == 3:
            ns = 0
        elif real_action == 1:
            ns = 4
        elif real_action == 2:
            ns = 1
    elif state == 1:
        if real_action == 0:
            ns = 0
        elif real_action == 1:
            ns = 5
        elif real_action == 2:
            ns = 2
        elif real_action == 3:
            ns = 1
    elif state == 3:
        if real_action == 0:
            ns = 2
        elif real_action == 1:
            ns = 6
        elif real_action == 2 or real_action == 3:
            ns = 3
    elif state == 4:
        if real_action == 0 or real_action == 1:
            ns = 4
        elif real_action == 2:
            ns = 5
        elif real_action == 3:
            ns = 0
    elif state == 5:
        if real_action == 0:
            ns = 4
        elif real_action == 1:
            ns = 7
        elif real_action == 2:
            ns = 5
        elif real_action == 3:
            ns = 1
    elif state == 6:
        if real_action == 0 or real_action == 2:
            ns = 6
        elif real_action == 1:
            ns = 9
        elif real_action == 3:
            ns = 3
    elif state == 7:
        if real_action == 0 or real_action == 1:
            ns = 7
        elif real_action == 2:
            ns = 8
        elif real_action == 3:
            ns = 5
    elif state == 8:
        if real_action == 0:
            ns = 7
        elif real_action == 1:
            ns = 10
        elif real_action == 2:
            ns = 9
        elif real_action == 3:
            ns = 8
    elif state == 9:
        if real_action == 0:
            ns = 8
        elif real_action == 1 or real_action == 2:
            ns = 9
        elif real_action == 3:
            ns = 6
    
    # evaluate reward and its termination
    if ns == 2:
        reward = 10
        done = True
    elif ns == 10:
        reward = -100
        done = True
    else:
        reward = -1
        done = False

    return ns, reward, done

def learn(Q, alpha, gamma, p, nb_episodes):
    print('Running the Q-Learning Algorithm for {} times'.format(nb_episodes))

    for i in range(nb_episodes):
        state = 0       # Init initial state of the agent
        done = False    # Set done to False, is required to enter the loop

        # The Q-Table learning algorithm with discounted reward
        while not done:
            # Choose an action which maximises the reward (from the current state)
            # When choosing an action, use decaying random noise for path exploration
            action = np.argmax(Q[state, :] + np.random.randn(1, len(Q[0])) / (i + 1))

            # Receive a feedback after taking the desired action with probability p
            new_state, reward, done = step(state, action, p)

            # In stochastic world, we need not to rely on previous Q-Table
            # This can be done by adding the learning rate , alpha
            Q[state, action] = (1 - alpha) * Q[state, action] \
                                + alpha * (reward + gamma * np.max(Q[new_state, :]))

            # Update state to the next one
            state = new_state
    
    return Q
        

if __name__ == '__main__':
    # Initialise Q-Table with zeros representing rewards
    nb_states = 11 
    nb_actions = 4 
    Q = np.zeros([nb_states, nb_actions])

    # Set learning rate, alpha
    alpha = .95

    # Set discount reward factor, gamma
    gamma = .35

    # Set probability of performing the desired action
    p = 0.5

    # Set number of iterations
    nb_episodes = 10000

    # Run the q-learning algorithm
    #Q = learn(Q, alpha, gamma, p, nb_episodes)

    # Print the result
    print('    Left         Down         Right        Up')
    print(Q)

    print(get_transition_matrix())

