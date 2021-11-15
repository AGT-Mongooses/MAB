import numpy as np
from tqdm import tqdm

# TODO remove/modify #DEBUG lines

def simulate(policy, pA, pB, s0, max_cars, price_rent, price_drive, steps):

    states = []
    actions = []
    rewards = []

    s = s0 #starting state; Morning
    print("Starting state: ", s) #DEBUG

    for i in range(steps):
        peopleA = np.random.choice(list(range(len(pA))), p=pA)
        print("People in A: ", peopleA)  # DEBUG
        peopleB = np.random.choice(list(range(len(pB))), p=pB)
        print("People in B: ", peopleB)  # DEBUG

        rentedA = min(s, peopleA)
        print("Rented in A: ", rentedA)  # DEBUG
        rentedB = min(max_cars - s, peopleB)
        print("Rented in B: ", rentedB)  # DEBUG

        r = price_rent * (rentedA + rentedB)
        print("Reward (no cost): ", r)

        s = s - rentedA + rentedB  # Evening
        states.append(s)
        print("Current state: ", s) #DEBUG

        a = policy(s)
        actions.append(a)
        print("Action by policy: ", a) #DEBUG

        s_next = s
        c = 0 #Case flag; 1 - include driving cost in reward; 0 - don't

        if a == 1: # A->B
            s_next -= 1
            c = 1
            if s_next < 0:
                print("Cannot do action", a, " from state ", s, "! Doing action 0 instead...") #DEBUG
                s_next = 0
                c = 0

        elif a == 2: #B->A
            s_next += 1
            c = 1
            if s_next > max_cars:
                print("Cannot do action", a, " from state ", s, "! Doing action 0 instead...") #DEBUG
                s_next = max_cars
                c = 0

        if c == 1:
            r -= price_drive
            print("Reward (cost adjusted): ", r) #DEBUG

        rewards.append(r)

        print("Next state: ", s_next) #DEBUG
        print("+++++++++++++++") #DEBUG
        s = s_next #Morning

    return np.asarray(states), np.asarray(actions), np.asarray(rewards)

def greedy_policy(s):
    # TODO Implement a greedy policy
    return np.random.choice([0, 1, 2])

def compute_action_values(states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, max_cars):
    print("COMPUTING ACTION VALUES") #DEBUG
    sa = np.zeros(shape = (max_cars + 1, 3)) # num_possible_states x num_possible_action matrix; Value of actions in states
    d =  np.zeros(shape = (max_cars + 1, 3)) #DEBUG ; keeps track of visited state-action combinations
    m = -1 * np.ones(shape = (max_cars + 1, len(states))) # num_possible_states x len(states); Stores the index of every occurrence of a state


    for s in range(max_cars + 1):
        #Find the state and store each occurrence
        j = -1 #Occurrence counter; -1 = No occurrences
        for i, state in enumerate(states):
            if state == s:
                j += 1
                m[s, j] = i
        #Go through all possible actions, find the first occurrence and compute the value
        if j >= 0:
            for a in range(3):
                for index in range(j):
                    if m[s, index] == a:
                        d[s, a] = 1 #DEBUG
                        sa[s, a] = np.sum(rewards[int(m[s, index]):])
                        break
    return sa, d

def estimate_action_values(policy, pA, pB, max_cars, price_rent, price_drive, steps, simulations):
    sa_sum = np.zeros(shape=(max_cars + 1, 3))

    for i in tqdm(range(simulations)):
        s0 = np.random.choice(list(range(max_cars + 1)))
        states, actions, rewards = simulate(policy, pA, pB, s0, max_cars, price_rent, price_drive, steps)
        sa, d = compute_action_values(states, actions, rewards, max_cars) #DEBUG (delete d)

        print("\nSTATE-ACTIONS\n", sa) #DEBUG
        print("\nVISITED COMBINATIONS\n", d) #DEBUG
        sa_sum += sa

    return sa_sum / simulations

if __name__ == "__main__":
    s0 = 3
    max_cars = 5
    #pA = np.array([0.1, 0.4, 0.3, 0.1, 0.1])
    #pB = np.array([0.6, 0.2, 0.1, 0.1, 0.0])
    pA = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    pB = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    price_rent = 10
    price_drive = 5
    steps = 100
    policy = greedy_policy
    simulations = 3

    #states, actions, rewards = simulate(policy, pA, pB, s0, max_cars, price_rent, price_drive, steps)

    #for i in range(len(states)):
    #    print(i, ". Step | State = ", states[i], " | Action = ", actions[i], " | Reward = ", rewards[i])

    sa = estimate_action_values(policy, pA, pB, max_cars, price_rent, price_drive, steps, simulations)

    print("\nAVERAGE STATE-ACTION\n", sa) #DEBUG