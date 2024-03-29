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

    return [TL, TD, TR, TU]

def get_true_action(p, action):
    action_space = list(range(4))
    if random.random() < p:
        return action
    else:
        action_space.remove(action)
        return random.choice(action_space)

def step(state, action, p, tms):
    # Evaluate true action from this stochastic world
    true_action = get_true_action(p, action)

    # Get corresponding transition matrix
    tm = tms[true_action]

    try:
        # Find next state when performing the true_action at current state
        next_state = list(tm[state]).index(1)
    except ValueError:
        print('current state: ', state)
        print('current action: ', true_action)
        print('current matrix: \n', tm)

    # Evaluate reward and its termination
    if next_state == 2:
        reward = 10
        done = True
    elif next_state == 10:
        reward = -100
        done = True
    else:
        reward = -1
        done = False

    return next_state, reward, done

def init_state():
    non_terminal_states = list(range(11))
    non_terminal_states.remove(2)
    non_terminal_states.remove(10)
    return random.choice(non_terminal_states)

def learn(Q, tms, alpha, gamma, p, nb_episodes):
    print('Running the Q-Learning Algorithm for {} times'.format(nb_episodes))

    for _ in range(nb_episodes):
        state = init_state()    # Init initial state of the agent (exploration)
        done = False            # Is required to enter the loop

        # The Q-Table learning algorithm with discounted reward
        while not done:
            # Choose an action which will maximise the state-action value func
            action = np.argmax(Q[state, :])

            # Receive a feedback after taking a desired action with prob p
            new_state, reward, done = step(state, action, p, tms)

            # Copy Q to check the value differnece
            prev_Q = np.copy(Q)

            # In stochastic world, we need not to rely on previous Q-Table
            # This can be done by adding the learning rate , alpha
            Q[state, action] = (
                    (1 - alpha) * Q[state, action] 
                    + alpha * (reward + gamma * np.max(Q[new_state, :]))
                )
            
            # If there is no value change in Q
            if np.allclose(prev_Q, Q, 1e-15):
                return Q

            # Update state to the next one
            state = new_state
    
    return Q

if __name__ == '__main__':
    # Initialise Q-Table with zeros representing rewards
    nb_states = 11 
    nb_actions = 4 
    Q = np.zeros([nb_states, nb_actions])

    # Get transition matrices
    tms = get_transition_matrices()

    # Validate the transition matrices
    assert len(tms) == 4

    # Set learning rate, alpha
    alpha = .95

    # Set discount reward factor, gamma
    gamma = .35

    # Set probability of performing the desired action
    p = .5

    # Set number of iterations
    nb_episodes = int(1e5)

    # Run the q-learning algorithm
    Q = learn(Q, tms, alpha, gamma, p, nb_episodes)

    # Print the result
    print('    Left         Down         Right        Up')
    print(Q)

