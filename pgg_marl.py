
# pip install deap

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random

# Set Random Seeds
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # Multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Step 1: Define the Environment with Reward Shaping and Heterogeneous Agents
# Define the PublicGoodsGame Environment
class PublicGoodsGame:
    def __init__(self, n_agents, multiplier, initial_resources, penalty_factor=0.5):
        self.n_agents = n_agents
        self.multiplier = multiplier
        self.penalty_factor = penalty_factor
        self.resources = np.array(initial_resources, dtype=np.float64)

    def step(self, contributions):
        total_contribution = np.sum(contributions)
        public_reward = self.multiplier * total_contribution / self.n_agents
        rewards = public_reward - contributions

        # Reward shaping: Penalize free-riding
        penalties = np.where(contributions == 0, -self.penalty_factor, 0)
        shaped_rewards = rewards + penalties

        self.resources += shaped_rewards
        return self.resources, shaped_rewards

    def reset(self):
        self.resources = np.array(self.initial_resources, dtype=np.float64)
        return self.resources

# Step 2: Define the RL Agent
class RLAgent(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(RLAgent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Sigmoid()  # Ensure output is in [0, 1]
        )

    def forward(self, state):
        return self.fc(state)

# Step 3: Initialize Environment and Agents
multiplier = 2 

initial_resources = [100, 120, 80, 110, 90]  # Heterogeneous initial resources
n_agents = len(initial_resources)
print("n_agents: ", n_agents)

penalty_factor = 3 
env = PublicGoodsGame(n_agents, multiplier, initial_resources, penalty_factor)

agents = [RLAgent(input_dim=2, action_dim=1) for _ in range(n_agents)]
optimizers = [optim.Adam(agent.parameters(), lr=0.01) for agent in agents]

# Step 4: Train the Agents
episodes = 500
payoff_results = []
contributions_over_time = []
cumulative_rewards = np.zeros((episodes, n_agents))
nash_equilibrium_turns = None # need for the sefl contailn code
optimal_cooperation_turns = None #

for episode in range(episodes):
    contributions = []
    for i, agent in enumerate(agents):
        # Enhanced state representation: Individual resources and global mean
        state = torch.tensor([env.resources[i], env.resources.mean()], dtype=torch.float32)
        action = agent(state).detach().numpy() + np.random.normal(0, 0.1)  # Add exploration noise
        contribution = np.clip(action, 0, 1).item()  # Restrict contributions to [0, 1]
        contributions.append(contribution)
    resources, rewards = env.step(contributions)

    # Update agents using the rewards
    for i, agent in enumerate(agents):
        optimizer = optimizers[i]
        optimizer.zero_grad()
        state = torch.tensor([env.resources[i], env.resources.mean()], dtype=torch.float32)
        predicted_action = agent(state)
        loss = (rewards[i] - predicted_action).pow(2).mean()  # Reward-linked loss
        loss.backward()
        optimizer.step()

    # Check for Nash equilibrium and optimal cooperation
    total_contribution = np.sum(contributions)
    if nash_equilibrium_turns is None and total_contribution == 0:  # All defect
        nash_equilibrium_turns = episode
    if optimal_cooperation_turns is None and total_contribution == n_agents:  # All cooperate
        optimal_cooperation_turns = episode

    # Record results for analysis
    contributions_over_time.append(contributions)
    payoff_results.append((episode, contributions, rewards))
    cumulative_rewards[episode] = cumulative_rewards[episode - 1] + rewards if episode > 0 else rewards

# Step 5: Visualization
episodes_list = list(range(episodes))
total_contributions = [np.sum(c) for c in contributions_over_time]

# Plot Total Contributions Over Time
plt.figure(figsize=(12, 6))
plt.plot(episodes_list, total_contributions, label='Total Contributions')
if nash_equilibrium_turns is not None:
    plt.axvline(nash_equilibrium_turns, color='red', linestyle='--', label='Nash Equilibrium')
if optimal_cooperation_turns is not None:
    plt.axvline(optimal_cooperation_turns, color='green', linestyle='--', label='Optimal Cooperation')
plt.xlabel('Episode')
plt.ylabel('Total Contributions')
plt.title('Contributions Over Time')
plt.legend()
plt.show()

# Plot Individual Contributions Over Time
plt.figure(figsize=(12, 6))
for i in range(n_agents):
    agent_contributions = [c[i] for c in contributions_over_time]
    plt.plot(episodes_list, agent_contributions, label=f'Agent {i+1} Contributions')
plt.xlabel('Episode')
plt.ylabel('Individual Contributions')
plt.title('Individual Contributions Over Time')
plt.legend()
plt.show()

# Plot Cumulative Rewards
plt.figure(figsize=(12, 6))
for i in range(n_agents):
    plt.plot(episodes_list, cumulative_rewards[:, i], label=f'Agent {i+1} Cumulative Rewards')
plt.xlabel('Episode')
plt.ylabel('Cumulative Rewards')
plt.title('Cumulative Rewards Over Time')
plt.legend()
plt.show()

# Step 6: Print Results
print(f"Nash Equilibrium achieved in {nash_equilibrium_turns} turns.")
print(f"Optimal Cooperation achieved in {optimal_cooperation_turns} turns.")

# Step 7: Evaluate Results
print("\nFinal Contributions and Resources:")
for i in range(n_agents):
    print(f"Agent {i+1}: Resources = {env.resources[i]:.2f}")

print("\nFinal Contributions and Payoffs by Episode:")
for episode, contributions, rewards in payoff_results[-10:]:  # Last 10 episodes
    print(f"Episode {episode}: Contributions = {contributions}, Rewards = {rewards}")



#------------------------------------------------------- Optimized ------------------------------------------------------#

def evaluate_penalty_factor(penalty_factor, n_agents=n_agents, multiplier=multiplier, initial_resources=initial_resources, episodes=episodes):

    # Initialize Environment and Agents
    env = PublicGoodsGame(n_agents, multiplier, initial_resources, penalty_factor)
    agents = [RLAgent(input_dim=2, action_dim=1) for _ in range(n_agents)]
    optimizers = [optim.Adam(agent.parameters(), lr=0.01) for agent in agents]

    contributions_over_time = []
    cumulative_rewards = np.zeros((episodes, n_agents))
    total_contributions = []
    optimal_cooperation_turns = None
    nash_equilibrium_turns = None

    for episode in range(episodes):
        contributions = []
        for i, agent in enumerate(agents):
            # Enhanced state representation
            state = torch.tensor([env.resources[i], env.resources.mean()], dtype=torch.float32)
            action = agent(state).detach().numpy() + np.random.normal(0, 0.1)  # Add exploration noise
            contribution = np.clip(action, 0, 1).item()
            contributions.append(contribution)

        # Environment processes contributions
        resources, rewards = env.step(contributions)
        contributions_over_time.append(contributions)
        cumulative_rewards[episode] = cumulative_rewards[episode - 1] + rewards if episode > 0 else rewards
        total_contributions.append(np.sum(contributions))

        # Update agents based on contributions
        for i, agent in enumerate(agents):
            optimizer = optimizers[i]
            optimizer.zero_grad()
            state = torch.tensor([env.resources[i], env.resources.mean()], dtype=torch.float32)
            predicted_action = agent(state)
            loss = (predicted_action - torch.tensor([contributions[i]], dtype=torch.float32)).pow(2).mean()
            loss.backward()
            optimizer.step()

        # Check for Nash equilibrium and optimal cooperation
        total_contribution = np.sum(contributions)
        if nash_equilibrium_turns is None and total_contribution == 0:  # All defect
            nash_equilibrium_turns = episode
        if optimal_cooperation_turns is None and total_contribution == n_agents:  # All cooperate
            optimal_cooperation_turns = episode

    return -np.mean(total_contributions), contributions_over_time, cumulative_rewards, env.resources, optimal_cooperation_turns

# Genetic Algorithm Optimization

def optimize_penalty_factor(n_agents=n_agents, multiplier=multiplier, initial_resources=initial_resources, episodes=episodes):

    # Define Fitness and Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Single-objective: minimize average contributions
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0.1, 10)  # Penalty factor range
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def fitness_function(individual):
        penalty_factor = individual[0]
        avg_contribution, _, _, _, _ = evaluate_penalty_factor(penalty_factor, n_agents, multiplier, initial_resources, episodes)
        return avg_contribution,

    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=20)
    ngen = 2  # Number of generations
    hof = tools.HallOfFame(1)

    # Run GA
    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=ngen,
        stats=None,
        halloffame=hof,
        verbose=True,
    )

    return hof[0][0]  # Best penalty factor

# Run Optimization
best_penalty_factor = optimize_penalty_factor()
print(f"Best Penalty Factor: {best_penalty_factor}")

# Re-run the simulation with the best penalty factor and plot results
_, contributions_over_time, cumulative_rewards, final_resources, optimal_cooperation_turns = evaluate_penalty_factor(best_penalty_factor)

# Visualization
# 1. Plot Total Contributions Over Time
plt.figure(figsize=(12, 6))
total_contributions = [np.sum(c) for c in contributions_over_time]
plt.plot(range(len(total_contributions)), total_contributions, label="Total Contributions")
if nash_equilibrium_turns is not None: #
    plt.axvline(nash_equilibrium_turns, color='red', linestyle='--', label='Nash Equilibrium') #
if optimal_cooperation_turns is not None: #
    plt.axvline(optimal_cooperation_turns, color='green', linestyle='--', label='Optimal Cooperation') #
plt.xlabel("Episode")
plt.ylabel("Total Contributions")
plt.title("Total Contributions Over Time")
plt.legend()
plt.grid()
plt.show()

# 2. Plot Individual Contributions Over Time
plt.figure(figsize=(12, 6))
for i in range(len(contributions_over_time[0])):
    agent_contributions = [c[i] for c in contributions_over_time]
    plt.plot(range(len(agent_contributions)), agent_contributions, label=f"Agent {i+1}")
plt.xlabel("Episode")
plt.ylabel("Individual Contributions")
plt.title("Individual Contributions Over Time")
plt.legend()
plt.grid()
plt.show()

# 3. Plot Cumulative Rewards Over Time
plt.figure(figsize=(12, 6))
for i in range(len(cumulative_rewards[0])):
    plt.plot(range(len(cumulative_rewards)), cumulative_rewards[:, i], label=f"Agent {i+1} Cumulative Rewards")
plt.xlabel("Episode")
plt.ylabel("Cumulative Rewards")
plt.title("Cumulative Rewards Over Time")
plt.legend()
plt.grid()
plt.show()

# Results
# Step 6: Print Results
print(f"Nash Equilibrium achieved in {optimal_cooperation_turns} turns.")
print(f"Optimal Cooperation achieved in {optimal_cooperation_turns} turns.")

# Step 7: Evaluate Results
print("\nFinal Contributions and Resources:")
for i in range(len(final_resources)):
    print(f"Agent {i+1}: Resources = {final_resources[i]:.2f}")



