#!/usr/bin/env python3
"""
Multi-run Restarting Bandit Experiment - Plot Only
Runs experiments multiple times and saves only the performance comparison plot
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from datetime import datetime

class RestartingBanditExperimentRunner:
    def __init__(self, results_dir="results", num_runs=5, experiment_name="restarting_bandit_comparison"):
        self.results_dir = Path(results_dir)
        self.num_runs = num_runs
        self.experiment_name = experiment_name
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"Running {num_runs} experiments...")
        
        # Environment Setup - Restarting Bandit from Paper
        self.states = [0, 1, 2, 3, 4]  # 5 states as in paper
        self.actions = [0, 1]  # 0: passive, 1: active
        self.a = 0.9  # Reward parameter
        
        # Transition matrices from paper
        # Passive mode: tendency to go up state space
        self.P0 = np.array([
            [0.1, 0.9, 0.0, 0.0, 0.0],  # State 0 -> mostly to state 1
            [0.1, 0.0, 0.9, 0.0, 0.0],  # State 1 -> mostly to state 2
            [0.1, 0.0, 0.0, 0.9, 0.0],  # State 2 -> mostly to state 3
            [0.1, 0.0, 0.0, 0.0, 0.9],  # State 3 -> mostly to state 4
            [0.1, 0.0, 0.0, 0.0, 0.9]   # State 4 -> mostly stays in state 4
        ])
        
        # Active mode: restart to state 0 with probability 1
        self.P1 = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],  # All transitions go to state 0
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0]
        ])
        
        # Exact Whittle indices from paper
        self.optimal_index = {0: -0.9, 1: -0.73, 2: -0.5, 3: -0.26, 4: -0.01}
        
        # Simulation parameters (will be updated in main)
        self.N = 5         # Total number of arms
        self.M = 1         # Number of arms that can be active
        self.T = 10000     # Total time steps
        self.gamma = 0.99  # Discount factor

    def get_reward(self, state, action):
        """Reward function: r(s,0) = a^s for passive, r(s,1) = 0 for active"""
        if action == 0:  # Passive mode
            return self.a ** state  # Exponential: a^k
        else:  # Active mode
            return 0.0

    def sample_next_state(self, s, a):
        """Sample next state based on action and current state"""
        if a == 1:  # Active mode - restart to state 0
            return 0
        else:  # Passive mode - upward drift
            probs = self.P0[s]
            return np.random.choice(self.states, p=probs)

    def simulate_optimal(self):
        """Optimal policy using true Whittle indices"""
        X = [random.choice(self.states) for _ in range(self.N)]
        cumulative_reward = 0
        cumulative_avg = []

        for t in range(1, self.T+1):
            priorities = [self.optimal_index[X[i]] for i in range(self.N)]
            active_arms = np.argsort(priorities)[-self.M:]
            A = [1 if i in active_arms else 0 for i in range(self.N)]

            step_reward = 0
            X_next = [None] * self.N
            for i in range(self.N):
                s = X[i]
                a = A[i]
                r = self.get_reward(s, a)
                step_reward += r
                X_next[i] = self.sample_next_state(s, a)

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)
            X = X_next.copy()

        return np.array(cumulative_avg)

    def simulate_two_timescale(self):
        """Two-timescale stochastic approximation algorithm"""
        X = [random.choice(self.states) for _ in range(self.N)]

        # Initialize Q-functions for all arms, states, actions, and target states
        Q = {}
        for arm in range(self.N):
            Q[arm] = {}
            for state in self.states:
                Q[arm][state] = {}
                for action in self.actions:
                    Q[arm][state][action] = {}
                    for target_state in self.states:
                        Q[arm][state][action][target_state] = 0.0

        # Initialize lambda estimates
        lambda_est = {state: 0.0 for state in self.states}

        cumulative_reward = 0
        cumulative_avg = []

        for t in range(1, self.T+1):
            # Epsilon-greedy exploration
            epsilon = 0.1
            if random.random() < epsilon:
                active_arms = random.sample(range(self.N), self.M)
            else:
                priorities = [lambda_est[X[i]] for i in range(self.N)]
                active_arms = np.argsort(priorities)[-self.M:]

            A = [1 if i in active_arms else 0 for i in range(self.N)]
            step_reward = 0
            X_next = [None] * self.N

            # Update Q-values and collect transitions
            for i in range(self.N):
                s = X[i]
                a = A[i]
                r = self.get_reward(s, a)
                step_reward += r
                s_next = self.sample_next_state(s, a)
                X_next[i] = s_next

                # Q-learning update with fixed learning rate
                alpha = 0.02
                for k in self.states:
                    old_q = Q[i][s][a][k]
                    current_lambda = lambda_est[k]
                    max_q_next = max(Q[i][s_next][v][k] for v in self.actions)
                    td_target = r - current_lambda * a + self.gamma * max_q_next
                    Q[i][s][a][k] = old_q + alpha * (td_target - old_q)

            # Update lambda estimates (slower timescale)
            beta = 0.005
            for k in self.states:
                q_active = np.mean([Q[i][k][1][k] for i in range(self.N)])
                q_passive = np.mean([Q[i][k][0][k] for i in range(self.N)])
                lambda_est[k] += beta * (q_active - q_passive)

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)
            X = X_next.copy()

        return np.array(cumulative_avg)

    def simulate_QWIC(self):
        """QWIC algorithm with grid search"""
        X = [random.choice(self.states) for _ in range(self.N)]
        lambda_grid = np.linspace(-1.25, 1.25, 10)

        # Initialize Q-values for each lambda value
        Q = {}
        for l_idx in range(len(lambda_grid)):
            Q[l_idx] = {}
            for arm in range(self.N):
                Q[l_idx][arm] = {}
                for state in self.states:
                    Q[l_idx][arm][state] = {0: 0.0, 1: 0.0}

        # Initialize Whittle indices
        whittle_indices = {}
        for arm in range(self.N):
            whittle_indices[arm] = {state: 0.0 for state in self.states}

        cumulative_reward = 0
        cumulative_avg = []

        def alpha_t(t):
            return min(0.1, 1.0 / np.sqrt(t))

        def epsilon_t(t):
            return 0.1 / np.sqrt(t)

        for t in range(1, self.T+1):
            learning_rate = alpha_t(t)
            exploration_rate = epsilon_t(t)

            # Select arms based on current Whittle indices
            current_whittle = []
            for arm in range(self.N):
                state = X[arm]
                whittle_val = whittle_indices[arm][state]
                current_whittle.append((whittle_val, arm))

            current_whittle.sort(reverse=True)
            active_arms = [arm for _, arm in current_whittle[:self.M]]

            # Add exploration
            if np.random.random() < exploration_rate:
                active_arms = np.random.choice(self.N, self.M, replace=False).tolist()

            A = [1 if i in active_arms else 0 for i in range(self.N)]
            step_reward = 0
            X_next = [None] * self.N

            # Execute actions and update Q-values
            for i in range(self.N):
                s = X[i]
                a = A[i]
                r = self.get_reward(s, a)
                step_reward += r
                X_next[i] = self.sample_next_state(s, a)

                # Update Q-values for current Whittle index
                current_whittle_val = whittle_indices[i][s]
                lambda_idx = np.argmin(np.abs(lambda_grid - current_whittle_val))

                old_q = Q[lambda_idx][i][s][a]
                max_q_next = max(Q[lambda_idx][i][X_next[i]][action] for action in self.actions)
                td_target = r - current_whittle_val * a + self.gamma * max_q_next
                new_q = (1 - learning_rate) * old_q + learning_rate * td_target
                Q[lambda_idx][i][s][a] = new_q

            # Update Whittle indices by finding lambda where Q^a = Q^p
            for arm in range(self.N):
                for state in self.states:
                    best_lambda_idx = 0
                    min_diff = float('inf')

                    for l_idx in range(len(lambda_grid)):
                        q_active = Q[l_idx][arm][state][1]
                        q_passive = Q[l_idx][arm][state][0]
                        diff = abs(q_active - q_passive)

                        if diff < min_diff:
                            min_diff = diff
                            best_lambda_idx = l_idx

                    whittle_indices[arm][state] = lambda_grid[best_lambda_idx]

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)
            X = X_next.copy()

        return np.array(cumulative_avg)

    def simulate_WIQL(self):
        """WIQL algorithm with visit-count based learning rates"""
        X = [random.choice(self.states) for _ in range(self.N)]
        Q = [{s: {a: 0.0 for a in self.actions} for s in self.states} for _ in range(self.N)]
        counts = [{s: {a: 0 for a in self.actions} for s in self.states} for _ in range(self.N)]
        lambda_est = [{s: 0.0 for s in self.states} for _ in range(self.N)]

        cumulative_reward = 0
        cumulative_avg = []

        for t in range(1, self.T+1):
            # Epsilon-greedy with decreasing exploration
            eps = max(0.05, self.N / (self.N + t))
            if random.random() < eps:
                active_arms = random.sample(range(self.N), self.M)
            else:
                priorities = [lambda_est[i][X[i]] for i in range(self.N)]
                active_arms = np.argsort(priorities)[-self.M:]

            A = [1 if i in active_arms else 0 for i in range(self.N)]
            step_reward = 0
            X_next = [None] * self.N

            for i in range(self.N):
                s = X[i]
                a = A[i]
                r = self.get_reward(s, a)
                step_reward += r
                counts[i][s][a] += 1
                alpha = 1.0 / counts[i][s][a]
                s_next = self.sample_next_state(s, a)
                X_next[i] = s_next
                max_q_next = max(Q[i][s_next].values())
                Q[i][s][a] = (1 - alpha) * Q[i][s][a] + alpha * (r + self.gamma * max_q_next)

            # Update Whittle index estimates
            for i in range(self.N):
                for s in self.states:
                    lambda_est[i][s] = Q[i][s][1] - Q[i][s][0]

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)
            X = X_next.copy()

        return np.array(cumulative_avg)

    def simulate_adaptive_WIQL_UCB(self):
        """Adaptive WIQL with UCB exploration"""
        X = [random.choice(self.states) for _ in range(self.N)]
        Q = np.zeros((self.N, len(self.states), len(self.actions)))
        counts = np.zeros((self.N, len(self.states), len(self.actions)))
        lambda_est = np.zeros((self.N, len(self.states)))

        cumulative_reward = 0
        cumulative_avg = []
        c = 1.0  # UCB exploration parameter

        for t in range(1, self.T+1):
            # Compute UCB values for each arm
            ucb_values = np.zeros(self.N)
            for i in range(self.N):
                s = X[i]
                ucb_action_values = np.zeros(len(self.actions))
                for a in range(len(self.actions)):
                    if counts[i, s, a] > 0:
                        exploration = c * np.sqrt(np.log(t + 1) / counts[i, s, a])
                    else:
                        exploration = c * np.sqrt(np.log(t + 1))
                    ucb_action_values[a] = Q[i, s, a] + exploration

                # Whittle index with UCB
                ucb_values[i] = ucb_action_values[1] - ucb_action_values[0]

            active_arms = np.argsort(ucb_values)[-self.M:]
            A = [1 if i in active_arms else 0 for i in range(self.N)]

            step_reward = 0
            X_next = [None] * self.N
            for i in range(self.N):
                s = X[i]
                a = A[i]
                r = self.get_reward(s, a)
                step_reward += r

                counts[i, s, a] += 1
                alpha = 1.0 / counts[i, s, a]

                s_next = self.sample_next_state(s, a)
                X_next[i] = s_next

                max_q_next = max(Q[i, s_next, 0], Q[i, s_next, 1])
                Q[i, s, a] = (1 - alpha) * Q[i, s, a] + alpha * (r + self.gamma * max_q_next)

                # Update Whittle index estimate
                lambda_est[i, s] = Q[i, s, 1] - Q[i, s, 0]

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)
            X = X_next.copy()

        return np.array(cumulative_avg)

    def run_single_experiment(self, run_id):
        """Run all algorithms for a single experiment"""
        print(f"  Run {run_id + 1}/{self.num_runs}")
        
        # Set random seed for reproducibility
        np.random.seed(42 + run_id)
        random.seed(42 + run_id)
        
        # Run all algorithms and return only the performance curves
        optimal_avg = self.simulate_optimal()
        two_timescale_avg = self.simulate_two_timescale()
        qwic_avg = self.simulate_QWIC()
        wiql_avg = self.simulate_WIQL()
        adaptive_wiql_avg = self.simulate_adaptive_WIQL_UCB()
        
        return {
            'optimal': optimal_avg,
            'two_timescale': two_timescale_avg,
            'qwic': qwic_avg,
            'wiql': wiql_avg,
            'adaptive_wiql': adaptive_wiql_avg
        }

    def run_all_experiments_and_plot(self):
        """Run all experiments and create the performance plot"""
        # Collect results from all runs
        all_curves = {
            'optimal': [],
            'two_timescale': [],
            'qwic': [],
            'wiql': [],
            'adaptive_wiql': []
        }
        
        for run_id in range(self.num_runs):
            curves = self.run_single_experiment(run_id)
            for alg in all_curves.keys():
                all_curves[alg].append(curves[alg])
        
        # Calculate average curves and standard deviations
        avg_curves = {}
        std_curves = {}
        for alg in all_curves.keys():
            avg_curves[alg] = np.mean(all_curves[alg], axis=0)
            std_curves[alg] = np.std(all_curves[alg], axis=0)
        
        # Create the plot
        self.create_performance_plot(avg_curves, std_curves)
        
        # Print final performance summary
        print(f"\nFinal Performance Summary (last 1000 steps, mean ± std):")
        final_window = 1000
        algorithms = ['optimal', 'two_timescale', 'qwic', 'wiql', 'adaptive_wiql']
        labels = ['Optimal', 'Two-Timescale', 'QWIC', 'WIQL', 'Adaptive WIQL-UCB']
        
        for alg, label in zip(algorithms, labels):
            final_means = [curve[-final_window:].mean() for curve in all_curves[alg]]
            final_mean = np.mean(final_means)
            final_std = np.std(final_means)
            print(f"  {label:18}: {final_mean:.6f} ± {final_std:.6f}")
        
        # Print problem characteristics
        print(f"\nRestarting Bandit Problem Characteristics:")
        print(f"- Passive mode: upward drift with rewards r(s,0) = {self.a}^s (exponential)")
        print(f"- Active mode: restart to state 0 with zero reward")
        print(f"- True Whittle indices: {self.optimal_index}")
        print(f"- Upper states give exponentially higher rewards but are rarely visited")
        print(f"- All Whittle indices are negative (passive mode preferred)")
        print(f"- Exploration challenge: high reward states are hard to reach")

    def create_performance_plot(self, avg_curves, std_curves):
        """Create and save the performance comparison plot"""
        plt.figure(figsize=(6, 4))
        
        algorithms = ['optimal', 'two_timescale', 'qwic', 'wiql', 'adaptive_wiql']
        colors = ['green', 'blue', 'red', 'purple', 'orange']
        labels = ['Optimal', 'WIQL-AB', 'WIQL-Fu', 'WIQL-BAVT', 'WIQL-UCB']
        linestyles = ['-', '-', '--', ':', '-.']
        
        for alg, color, label, style in zip(algorithms, colors, labels, linestyles):
            mean_curve = avg_curves[alg]
            std_curve = std_curves[alg]
            
            plt.plot(mean_curve, label=label, color=color, linewidth=2.5, linestyle=style)
            
            # Add confidence intervals for learning algorithms (not optimal)
            if alg != 'optimal':
                plt.fill_between(range(len(mean_curve)), 
                               mean_curve - std_curve, 
                               mean_curve + std_curve, 
                               alpha=0.2, color=color)
        
        plt.xlabel("Time Step", fontsize=16)
        plt.ylabel("Cumulative Average Reward", fontsize=16)
        #plt.title(f"Algorithm Comparison ({self.num_runs} runs, N={self.N}, M={self.M}, a={self.a})", fontsize=16)
        plt.legend(fontsize=16)
        plt.tick_params(axis='both', labelsize=16)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, self.T)
        plt.tight_layout()
        
        # Create descriptive filename: experiment_name_N_M_a_T_runs.png
        filename = f"{self.experiment_name}_N{self.N}_M{self.M}_a{self.a}_T{self.T}_runs{self.num_runs}.png"
        plot_path = self.results_dir / filename
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nPlot saved to: {plot_path}")
        return plot_path


def main():
    parser = argparse.ArgumentParser(description='Run restarting bandit algorithm comparison (plot only)')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs (default: 5)')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory (default: results)')
    parser.add_argument('--time-steps', type=int, default=10000, help='Time steps per run (default: 10000)')
    parser.add_argument('--arms', type=int, default=5, help='Number of arms (default: 5)')
    parser.add_argument('--active-arms', type=int, default=1, help='Number of active arms per step (default: 1)')
    parser.add_argument('--reward-param', type=float, default=0.9, help='Reward parameter a (default: 0.9)')
    parser.add_argument('--name', type=str, default='restarting_bandit_comparison', help='Experiment name (default: restarting_bandit_comparison)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("RESTARTING BANDIT ALGORITHM COMPARISON")
    print("=" * 70)
    print(f"Experiment: {args.name}")
    print(f"Runs: {args.runs} | Time steps: {args.time_steps}")
    print(f"Arms: {args.arms} | Active arms: {args.active_arms} | Reward param: {args.reward_param}")
    print("Algorithms: Optimal, WIQL-AB, WIQL-Fu, WIQL-BAVT, WIQL-UCB")
    print("=" * 70)
    
    # Initialize experiment runner with custom name
    runner = RestartingBanditExperimentRunner(
        results_dir=args.results_dir, 
        num_runs=args.runs,
        experiment_name=args.name
    )
    
    # Update parameters
    runner.T = args.time_steps
    runner.N = args.arms
    runner.M = args.active_arms
    runner.a = args.reward_param
    
    # Run experiments and create plot
    runner.run_all_experiments_and_plot()
    
    print("=" * 70)
    print(f"EXPERIMENT COMPLETED!")
    print(f"Filename: {args.name}_N{args.arms}_M{args.active_arms}_a{args.reward_param}_T{args.time_steps}_runs{args.runs}.png")
    print("=" * 70)


if __name__ == "__main__":
    main()