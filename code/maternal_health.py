#!/usr/bin/env python3
"""
Health Intervention Study - 52 Week Multi-Algorithm Comparison
Simulates one year of health worker interventions for beneficiary engagement
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import time

class HealthInterventionStudy:
    def __init__(self, results_dir="results", num_runs=5, experiment_name="health_intervention_52weeks"):
        self.results_dir = Path(results_dir)
        self.num_runs = num_runs
        self.experiment_name = experiment_name
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"Health Intervention Study - 52 Week Simulation")
        
        # Environment Setup - Three-State Health Engagement Model
        self.states = [0, 1, 2]  # L (Lost cause), P (Persuadable), S (Self-motivated)
        self.actions = [0, 1]    # 0: no intervention, 1: intervention (health worker visit)
        self.state_rewards = [0, 1, 2]  # Rewards: L=0, P=1, S=2
        
        # Simulation parameters
        self.N = 5000    # Total number of beneficiaries (scaled up)
        self.M = 1000    # Number of interventions available per week (20% intervention rate)
        self.T = 160      # Total weeks (one year)
        self.gamma = 0.95  # Discount factor (weekly intervals)
        
        # Assign beneficiary categories: 0 = A (high improvement), 1 = B (medium), 2 = C (low)
        # Category A: 1000 arms (20%), Category B: 1000 arms (20%), Category C: 3000 arms (60%)
        self.arm_categories = [0]*(self.N//5) + [1]*(self.N//5) + [2]*(self.N - 2*(self.N//5))
        
       
    def sample_next_state(self, s, a, beneficiary_idx):
        """Sample next state based on intervention and beneficiary category"""
        category = self.arm_categories[beneficiary_idx]
        
        if a == 1:  # Intervention provided
            if category == 0:    # High improvement category
                probs = self.P_A_active[s]
            elif category == 1:  # Medium improvement category
                probs = self.P_B_active[s]
            else:                # Low improvement category (category == 2)
                probs = self.P_C_active[s]
        else:  # No intervention
            if category == 0:
                probs = self.P_A_passive[s]
            elif category == 1:
                probs = self.P_B_passive[s]
            else:
                probs = self.P_C_passive[s]
        
        return np.random.choice(self.states, p=probs)

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

        C, C_prime = 1.0, 0.1

        def alpha_t(n):
            return C / max(1, int(n / 10))  # Adjusted for weekly timescale

        def beta_t(n):
            if n % 5 == 0:  # Update every 5 weeks
                return C_prime / (1 + int(n * np.log(max(2, n)) / 10))
            return 0

        for t in range(1, self.T+1):
            if random.random() < 0.05:  # 5% exploration
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
                s_next = self.sample_next_state(s, a, i)
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
        lambda_grid = np.linspace(-2.0, 2.0, 15)  # Expanded grid for health context
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
            return min(0.5 * t**(-0.6), 0.8)  # Slower decay for longer horizon

        def epsilon_t(t):
            return max(0.05, 0.5 * t**(-0.5))  # Minimum 5% exploration

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
                X_next[i] = self.sample_next_state(s, a, i)

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
            eps = max(0.1, self.N / (self.N + t))  # Minimum 10% exploration
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
                s_next = self.sample_next_state(s, a, i)
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
        c = 2.0  # UCB exploration parameter

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

                s_next = self.sample_next_state(s, a, i)
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
        print(f"  Week-by-week simulation {run_id + 1}/{self.num_runs}")
        
        # Set random seed for reproducibility
        np.random.seed(42 + run_id)
        random.seed(42 + run_id)
        
        start_time = time.time()
        
        # Run all algorithms
        two_timescale_avg = self.simulate_two_timescale()
        qwic_avg = self.simulate_QWIC()
        wiql_avg = self.simulate_WIQL()
        adaptive_wiql_avg = self.simulate_adaptive_WIQL_UCB()
        
        elapsed = time.time() - start_time
        print(f"    Completed in {elapsed:.2f} seconds")
        
        return {
            'two_timescale': two_timescale_avg,
            'qwic': qwic_avg,
            'wiql': wiql_avg,
            'adaptive_wiql': adaptive_wiql_avg
        }

    def run_all_experiments_and_plot(self):
        """Run all experiments and create the performance plot"""
        print(f"\nRunning {self.num_runs} year-long health intervention studies...")
        total_start = time.time()
        
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
        
        total_elapsed = time.time() - total_start
        print(f"\nAll simulations completed in {total_elapsed:.2f} seconds")
        
        # Calculate average curves and standard deviations
        avg_curves = {}
        std_curves = {}
        for alg in all_curves.keys():
            avg_curves[alg] = np.mean(all_curves[alg], axis=0)
            std_curves[alg] = np.std(all_curves[alg], axis=0)
        
        # Create the plot
        self.create_performance_plot(avg_curves, std_curves)
        
        # Print final performance summary
        print(f"\nFinal Health Engagement Performance (Week 52, mean ± std):")
        algorithms = ['two_timescale', 'qwic', 'wiql', 'adaptive_wiql']
        labels = ['Two-Timescale', 'QWIC', 'WIQL', 'Adaptive WIQL-UCB']
        
        for alg, label in zip(algorithms, labels):
            final_means = [curve[-1] for curve in all_curves[alg]]
            final_mean = np.mean(final_means)
            final_std = np.std(final_means)
            print(f"  {label:<18}: {final_mean:.4f} ± {final_std:.4f}")
        
        # Print study characteristics
        print(f"\nHealth Intervention Study Characteristics:")
        print(f"- Duration: 52 weeks (1 year)")
        print(f"- Population: {self.N:,} beneficiaries")
        print(f"- Weekly interventions: {self.M:,} health worker visits ({self.M/self.N*100:.1f}%)")
        print(f"- Categories: {len([x for x in self.arm_categories if x==0]):,} high-response, {len([x for x in self.arm_categories if x==1]):,} medium-response, {len([x for x in self.arm_categories if x==2]):,} low-response")
        print(f"- States: L (Lost, <5% engagement), P (Persuadable, 5-50%), S (Self-motivated, >50%)")
        print(f"- Rewards: {self.state_rewards} for engagement levels")

    def create_performance_plot(self, avg_curves, std_curves):
        """Create and save the performance comparison plot"""
        plt.figure(figsize=(6, 4))
        
        algorithms = ['two_timescale', 'qwic', 'wiql', 'adaptive_wiql']
        colors = ['blue', 'red', 'purple', 'orange']
        labels = ['WIQL-AB', 'WIQL-Fu', 'WIQL-BAVT', 'WIQL-UCB']
        linestyles = ['-', '--', ':', '-.']
        
        weeks = range(1, self.T + 1)
        
        for alg, color, label, style in zip(algorithms, colors, labels, linestyles):
            mean_curve = avg_curves[alg]
            std_curve = std_curves[alg]
            
            plt.plot(weeks, mean_curve, label=label, color=color, linewidth=2.5, linestyle=style)
            
            # Add confidence intervals
            plt.fill_between(weeks, 
                           mean_curve - std_curve, 
                           mean_curve + std_curve, 
                           alpha=0.2, color=color)
        
        plt.xlabel("Week", fontsize=14)
        plt.ylabel("Cumulative Average Reward", fontsize=14)
        #plt.title(f"52-Week Health Intervention Study (N={self.N:,}, {self.M:,} weekly interventions)", fontsize=14)
        plt.legend( fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim(1, self.T)
        plt.tight_layout()
        
        # Create filename
        filename = f"{self.experiment_name}_N{self.N}_M{self.M}_T{self.T}weeks_runs{self.num_runs}.png"
        plot_path = self.results_dir / filename
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nPlot saved to: {plot_path}")
        return plot_path


def main():
    parser = argparse.ArgumentParser(description='Health Intervention Study - 52 Week Algorithm Comparison')
    parser.add_argument('--runs', type=int, default=5, help='Number of simulation runs (default: 5)')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory (default: results)')
    parser.add_argument('--beneficiaries', type=int, default=5000, help='Number of beneficiaries (default: 5000)')
    parser.add_argument('--interventions', type=int, default=1000, help='Weekly interventions (default: 1000)')
    parser.add_argument('--name', type=str, default='health_intervention_52weeks', help='Experiment name')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HEALTH INTERVENTION STUDY - 52 WEEK ALGORITHM COMPARISON")
    print("=" * 80)
    print(f"Study: {args.name}")
    print(f"Simulation runs: {args.runs} | Duration: 52 weeks (1 year)")
    print(f"Beneficiaries: {args.beneficiaries:,} | Weekly interventions: {args.interventions:,}")
    print("Categories: High-response (20%), Medium-response (20%), Low-response (60%)")
    print("Algorithms: Two-Timescale, QWIC, WIQL, Adaptive WIQL-UCB")
    print("=" * 80)
    
    # Initialize study
    study = HealthInterventionStudy(
        results_dir=args.results_dir, 
        num_runs=args.runs,
        experiment_name=args.name
    )
    
    # Update parameters
    study.N = args.beneficiaries
    study.M = args.interventions
    
    # Recalculate arm categories with new N
    study.arm_categories = [0]*(study.N//5) + [1]*(study.N//5) + [2]*(study.N - 2*(study.N//5))
    
    # Run study
    study.run_all_experiments_and_plot()
    
    print("=" * 80)
    print("HEALTH INTERVENTION STUDY COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()