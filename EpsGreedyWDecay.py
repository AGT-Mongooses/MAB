# Multi-Armed Bandit: Epsilon Greedy solution (with decay)
# Nadja Mandic, Nikola Djurovic, Nedeljko Tesanovic


# Library imports
import numpy as np
import matplotlib.pyplot as plt

# Slot machine (One armed bandit)
class SlotMachine:
    def __init__(self, min, max):
        self.min = min  # Minimum return
        self.max = max  # Maximum return
        self.pulls = 0  # Number of pulls
        self.sum = 0  # Cumulative return
        self.avg = 0  # Average return

    # Pull the bandit's arm
    def pull(self):
        r = np.random.randint(self.min, self.max)
        self.sum += r
        self.pulls += 1
        self.avg = self.sum / self.pulls
        return r


# Initial setup
# Slot machines (Bandits)
machines = [SlotMachine(-200, 100), SlotMachine(-5, 10), SlotMachine(-2, 5)]
# Epsilon
eps = 1 / len(machines)
# Decay rate
decay = 0.999
# Number of pulls
N = 15000
# Number of experimentation passes (Go through all machines and learn their behaviour)
n_exp = 100

# Simulation function
def simulate(machines, eps, decay, N, n_exp):
    total = np.zeros(N, dtype=int)   # Cumulative return (For the plot)
    sum = 0
    eps_pl = np.ones(N) * eps
    reward = np.zeros(N, dtype=int) #Return
    choice = np.zeros(N, dtype=int) #Index of selected machine
    best = 0  # Greediness target
    # Exploration
    # Go through every machine n_exp times
    for i in range(0, n_exp):
        for j in range(0, len(machines)):
            reward[[i * len(machines) + j]] = machines[j].pull()
            sum += reward[[i * len(machines) + j]]
            total[i * len(machines) + j] = sum
            choice[i * len(machines) + j] = j
            if machines[j].avg > machines[best].avg:
                best = j
            eps *= decay
            eps_pl[i * len(machines)+ j] = eps
    # Exploitation
    # Choose the best machine with (1-epsilon) probability
    for i in range(n_exp * len(machines), N):
        probs = np.ones(len(machines)) * (eps / (len(machines) - 1))
        probs[best] = 1 - eps
        j = np.random.choice(range(0, len(machines)), p=probs)
        reward[i] = machines[j].pull()
        sum += reward[i]
        total[i] = sum
        choice[i] = j
        if machines[j].avg > machines[best].avg:
            best = j
        eps *= decay
        eps_pl[i] = eps
    # Plot
    plt.subplot(311)
    plt.grid()
    plt.title("Cumulative reward")
    plt.plot(total, color='green')
    plt.subplot(312)
    plt.title("Reward per pull")
    plt.grid()
    plt.plot(reward, color='blue', label='True')
    plt.plot(np.ones(N) * (sum/N), color='red', label='Average')
    plt.legend(loc="upper right")
    plt.subplot(313)
    plt.grid()
    plt.title("Chosen machine")
    plt.plot(choice+1, 'o', label='Machine number')
    plt.plot(eps_pl, color='red', label='Epsilon')
    plt.legend(loc="upper right")
    plt.show()

#Run simulation
simulate(machines, eps, decay, N, n_exp)
