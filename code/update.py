#!/usr/bin/env python3
"""
Multi-run Heterogeneous Arms Bandit Experiment - Plot Only
Runs experiments multiple times and saves only the performance comparison plot
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from datetime import datetime

class HeterogeneousBanditExperimentRunner:
    def __init__(self, results_dir="results", num_runs=5, experiment_name="heterogeneous_bandit_comparison"):
        self.results_dir = Path(results_dir)
        self.num_runs = num_runs
        self.experiment_name = experiment_name
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"Running {num_runs} experiments...")
        
        # Environment Setup
        self.states = [0, 1, 2, 3, 4]
        self.actions = [0, 1]
        self.state_rewards = [0, -1, -2, -3, -4]
        
        # Simulation parameters (will be updated in main)
        self.N = 120  # Total number of arms (must be divisible by 3)
        self.M = 5    # Number of arms that can be active
        self.T = 10000  # Total time steps
        self.gamma = 0.9  # Discount factor
        
        # Assign groups: 0 = A, 1 = B, 2 = C
        self.arm_categories = [0]*(self.N//3) + [1]*(self.N//3) + [2]*(self.N//3)
        
        # Passive transition matrices for groups A, B, C
        self.P_A_passive = np.array([
            [0.6, 0.4, 0.0, 0.0, 0.0],  # State 0
            [0.0, 0.6, 0.4, 0.0, 0.0],  # State 1
            [0.0, 0.0, 0.6, 0.4, 0.0],  # State 2
            [0.0, 0.0, 0.0, 0.6, 0.4],  # State 3
            [0.0, 0.0, 0.0, 0.0, 1.0]   # State 4
        ])
        
        self.P_B_passive = np.array([
            [0.9, 0.1, 0.0, 0.0, 0.0],  # State 0
            [0.0, 0.9, 0.1, 0.0, 0.0],  # State 1
            [0.0, 0.0, 0.9, 0.1, 0.0],  # State 2
            [0.0, 0.0, 0.0, 0.9, 0.1],  # State 3
            [0.0, 0.0, 0.0, 0.0, 1.0]   # State 4
        ])
        
        self.P_C_passive = np.array([
            [0.5, 0.5, 0.0, 0.0, 0.0],  # State 0
            [0.0, 0.5, 0.5, 0.0, 0.0],  # State 1
            [0.0, 0.0, 0.5, 0.5, 0.0],  # State 2
            [0.0, 0.0, 0.0, 0.5, 0.5],  # State 3
            [0.0, 0.0, 0.0, 0.0, 1.0]   # State 4
        ])
        
        self.P_active = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],  # State 0
            [0.9, 0.1, 0.0, 0.0, 0.0],  # State 1
            [0.9, 0.0, 0.1, 0.0, 0.0],  # State 2
            [0.9, 0.0, 0.0, 0.1, 0.0],  # State 3
            [0.9, 0.0, 0.0, 0.0, 0.1]   # State 4
        ])

    def sample_next_state(self, s, a, arm_idx, t):
        """Dynamic transition function with switching at T//2"""
        if a == 1:
            return np.random.choice(5, p=self.P_active[s])

        group = self.arm_categories[arm_idx]

        # Switch dynamics at halfway point (currently disabled with t >= T)
        if t >= self.T:  # This condition means switching never happens during simulation
            if group == 0:  # Group A now follows B's dynamics
                return np.random.choice(5, p=self.P_B_passive[s])
            elif group == 1:  # Group B now follows A's dynamics
                return np.random.choice(5, p=self.P_A_passive[s])
            elif group == 2:  # Group C stays the same
                return np.random.choice(5, p=self.P_C_passive[s])
        else:  # First half - original dynamics
            if group == 0:
                return np.random.choice(5, p=self.P_A_passive[s])
            elif group == 1:
                return np.random.choice(5, p=self.P_B_passive[s])
            elif group == 2:
                return np.random.choice(5, p=self.P_C_passive[s])

        raise ValueError("Unknown group")

    def simulate_two_timescale(self):
        """Two-timescale stochastic approximation algorithm"""
        X = [random.choice(self.states) for _ in range(self.N)]
        Q = {}
        for arm in range(self.N):
            Q[arm] = {}
            for state in self.states:
                Q[arm][state] = {}
                for action in self.actions:
                    Q[arm][state][action] = {}
                    for target_state in self.states:
                        Q[arm][state][action][target_state] = 0.0

        lambda_est = {state: 0.0 for state in self.states}
        local_clocks = {}
        for state in self.states:
            local_clocks[state] = {}
            for action in self.actions:
                local_clocks[state][action] = 0

        cumulative_reward = 0
        cumulative_avg = []

        C, C_prime = 1, 0.1

        def alpha_t(n):
            return C / max(1, int(n / 500))

        def beta_t(n):
            if n % self.N == 0:
                return C_prime / (1 + int(n * np.log(max(2, n)) / 500))
            return 0

        for t in range(1, self.T+1):
            if random.random() < 0.01:
                active_arms = random.sample(range(self.N), self.M)
            else:
                priorities = [lambda_est[X[i]] for i in range(self.N)]
                active_arms = np.argsort(priorities)[-self.M:]

            A = [1 if i in active_arms else 0 for i in range(self.N)]
            step_reward = 0
            X_next = [None] * self.N

            for i in range(self.N):
                s, a = X[i], A[i]
                r = self.state_rewards[s]
                step_reward += r
                s_next = self.sample_next_state(s, a, i, t)
                X_next[i] = s_next

                local_clocks[s][a] += 1
                alpha = alpha_t(local_clocks[s][a])

                for k in self.states:
                    old_q = Q[i][s][a][k]
                    current_lambda = lambda_est[k]
                    max_q_next = max(Q[i][s_next][v][k] for v in self.actions)
                    td_target = r - current_lambda * a + self.gamma * max_q_next
                    Q[i][s][a][k] = old_q + alpha * (td_target - old_q)

            beta = beta_t(t)
            if beta > 0:
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
        Q = {}
        for l_idx in range(len(lambda_grid)):
            Q[l_idx] = {}
            for arm in range(self.N):
                Q[l_idx][arm] = {}
                for state in self.states:
                    Q[l_idx][arm][state] = {0: 0.0, 1: 0.0}

        whittle_indices = {}
        for arm in range(self.N):
            whittle_indices[arm] = {state: 0.0 for state in self.states}

        cumulative_reward = 0
        cumulative_avg = []

        def alpha_t(t):
            return min(2 * t**(-0.5), 1)

        def epsilon_t(t):
            return t**(-0.5)

        for t in range(1, self.T+1):
            learning_rate = alpha_t(t)
            exploration_rate = epsilon_t(t)

            current_whittle = []
            for arm in range(self.N):
                state = X[arm]
                whittle_val = whittle_indices[arm][state]
                current_whittle.append((whittle_val, arm))

            current_whittle.sort(reverse=True)
            active_arms = [arm for _, arm in current_whittle[:self.M]]

            if np.random.random() < exploration_rate:
                active_arms = np.random.choice(self.N, self.M, replace=False).tolist()

            A = [1 if i in active_arms else 0 for i in range(self.N)]
            step_reward = 0
            X_next = [None] * self.N

            for i in range(self.N):
                s, a = X[i], A[i]
                r = self.state_rewards[s]
                step_reward += r
                X_next[i] = self.sample_next_state(s, a, i, t)

                current_whittle_val = whittle_indices[i][s]
                lambda_idx = np.argmin(np.abs(lambda_grid - current_whittle_val))

                old_q = Q[lambda_idx][i][s][a]
                max_q_next = max(Q[lambda_idx][i][X_next[i]][action] for action in self.actions)
                td_target = r - current_whittle_val * a + self.gamma * max_q_next
                new_q = (1 - learning_rate) * old_q + learning_rate * td_target
                Q[lambda_idx][i][s][a] = new_q

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
            eps = self.N / (self.N + t)
            if random.random() < eps:
                active_arms = random.sample(range(self.N), self.M)
            else:
                priorities = [lambda_est[i][X[i]] for i in range(self.N)]
                active_arms = np.argsort(priorities)[-self.M:]

            A = [1 if i in active_arms else 0 for i in range(self.N)]
            step_reward = 0
            X_next = [None] * self.N

            for i in range(self.N):
                s, a = X[i], A[i]
                r = self.state_rewards[s]
                step_reward += r
                counts[i][s][a] += 1
                alpha = 1.0 / counts[i][s][a]
                s_next = self.sample_next_state(s, a, i, t)
                X_next[i] = s_next
                max_q_next = max(Q[i][s_next].values())
                Q[i][s][a] = (1 - alpha) * Q[i][s][a] + alpha * (r + self.gamma * max_q_next)

            for i in range(self.N):
                s = X[i]
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
        c = 1.0

        for t in range(1, self.T+1):
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

                ucb_values[i] = ucb_action_values[1] - ucb_action_values[0]

            active_arms = np.argsort(ucb_values)[-self.M:]
            A = [1 if i in active_arms else 0 for i in range(self.N)]

            step_reward = 0
            X_next = [None] * self.N
            for i in range(self.N):
                s, a = X[i], A[i]
                r = self.state_rewards[s]
                step_reward += r

                counts[i, s, a] += 1
                alpha = 1.0 / counts[i, s, a]

                s_next = self.sample_next_state(s, a, i, t)
                X_next[i] = s_next

                max_q_next = max(Q[i, s_next, 0], Q[i, s_next, 1])
                Q[i, s, a] = (1 - alpha) * Q[i, s, a] + alpha * (r + self.gamma * max_q_next)

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
        
        # Update arm categories based on current N
        self.arm_categories = [0]*(self.N//3) + [1]*(self.N//3) + [2]*(self.N//3)
        
        # Run all algorithms and return only the performance curves
        two_timescale_avg = self.simulate_two_timescale()
        qwic_avg = self.simulate_QWIC()
        wiql_avg = self.simulate_WIQL()
        adaptive_wiql_avg = self.simulate_adaptive_WIQL_UCB()
        
        return {
            'two_timescale': two_timescale_avg,
            'qwic': qwic_avg,
            'wiql': wiql_avg,
            'adaptive_wiql': adaptive_wiql_avg
        }

    def run_all_experiments_and_plot(self):
        """Run all experiments and create the performance plot"""
        # Collect results from all runs
        all_curves = {
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
        algorithms = ['two_timescale', 'qwic', 'wiql', 'adaptive_wiql']
        labels = ['WIQL-AB', 'WIQL-Fu', 'WIQL-BAVT', 'WIQL-UCB']
        
        for alg, label in zip(algorithms, labels):
            final_means = [curve[-final_window:].mean() for curve in all_curves[alg]]
            final_mean = np.mean(final_means)
            final_std = np.std(final_means)
            print(f"  {label:12}: {final_mean:.6f} ± {final_std:.6f}")
        
        # Print problem characteristics
        print(f"\nHeterogeneous Arms Problem Characteristics:")
        print(f"- 3 arm groups (A, B, C) with different transition dynamics")
        print(f"- Group A: moderate deterioration (P=0.6 stay, 0.4 worsen)")
        print(f"- Group B: slow deterioration (P=0.9 stay, 0.1 worsen)")
        print(f"- Group C: fast deterioration (P=0.5 stay, 0.5 worsen)")
        print(f"- Active action: helps reset state with high probability")
        print(f"- Negative rewards: [0, -1, -2, -3, -4] for states [0, 1, 2, 3, 4]")

    def create_performance_plot(self, avg_curves, std_curves):
        """Create and save the performance comparison plot"""
        plt.figure(figsize=(6, 4))
        
        algorithms = ['two_timescale', 'qwic', 'wiql', 'adaptive_wiql']
        colors = ['blue', 'red', 'purple', 'orange']
        labels = ['WIQL-AB', 'WIQL-Fu', 'WIQL-BAVT', 'WIQL-UCB']
        linestyles = ['-', '--', ':', '-.']
        
        for alg, color, label, style in zip(algorithms, colors, labels, linestyles):
            mean_curve = avg_curves[alg]
            std_curve = std_curves[alg]
            
            plt.plot(mean_curve, label=label, color=color, linewidth=2.5, linestyle=style)
            
            # Add confidence intervals
            plt.fill_between(range(len(mean_curve)), 
                           mean_curve - std_curve, 
                           mean_curve + std_curve, 
                           alpha=0.2, color=color)
        
        plt.xlabel("Time Step", fontsize=16)
        plt.ylabel("Cumulative Average Reward", fontsize=16)
        #plt.title(f"Heterogeneous Arms Problem ({self.num_runs} runs, N={self.N}, M={self.M})", fontsize=16)
        plt.legend(loc='lower right', fontsize=14)
        plt.tick_params(axis='both', labelsize=16)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, self.T)
        plt.tight_layout()
        
        # Create descriptive filename: experiment_name_N_M_T_runs.png
        filename = f"{self.experiment_name}_N{self.N}_M{self.M}_T{self.T}_runs{self.num_runs}.png"
        plot_path = self.results_dir / filename
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nPlot saved to: {plot_path}")
        return plot_path


def main():
    parser = argparse.ArgumentParser(description='Run heterogeneous arms bandit algorithm comparison (plot only)')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs (default: 5)')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory (default: results)')
    parser.add_argument('--time-steps', type=int, default=10000, help='Time steps per run (default: 10000)')
    parser.add_argument('--arms', type=int, default=6, help='Number of arms - must be divisible by 3 (default: 120)')
    parser.add_argument('--active-arms', type=int, default=5, help='Number of active arms per step (default: 5)')
    parser.add_argument('--name', type=str, default='heterogeneous_bandit_comparison', help='Experiment name (default: heterogeneous_bandit_comparison)')
    
    args = parser.parse_args()
    
    # Ensure N is divisible by 3
    if args.arms % 3 != 0:
        print(f"Warning: Number of arms ({args.arms}) not divisible by 3. Adjusting to {args.arms - (args.arms % 3)}")
        args.arms = args.arms - (args.arms % 3)
    
    print("=" * 70)
    print("HETEROGENEOUS ARMS BANDIT ALGORITHM COMPARISON")
    print("=" * 70)
    print(f"Experiment: {args.name}")
    print(f"Runs: {args.runs} | Time steps: {args.time_steps}")
    print(f"Arms: {args.arms} ({args.arms//3} per group) | Active arms: {args.active_arms}")
    print("Groups: A (moderate), B (slow), C (fast deterioration)")
    print("Algorithms: WIQL-AB, WIQL-Fu, WIQL-BAVT, WIQL-UCB")
    print("=" * 70)
    
    # Initialize experiment runner with custom name
    runner = HeterogeneousBanditExperimentRunner(
        results_dir=args.results_dir, 
        num_runs=args.runs,
        experiment_name=args.name
    )
    
    # Update parameters
    runner.T = args.time_steps
    runner.N = args.arms
    runner.M = args.active_arms
    
    # Run experiments and create plot
    runner.run_all_experiments_and_plot()
    
    print("=" * 70)
    print(f"EXPERIMENT COMPLETED!")
    print(f"Filename: {args.name}_N{args.arms}_M{args.active_arms}_T{args.time_steps}_runs{args.runs}.png")
    print("=" * 70)


if __name__ == "__main__":
    main()