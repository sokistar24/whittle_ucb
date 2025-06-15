#!/usr/bin/env python3
"""
Sensor Policy Comparison Script with Multi-Run Averaging
Compares WIQL-UCB, Round Robin (RR), and Age of Information (AoI) policies
for sensor scheduling using temperature data with statistical reliability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from pathlib import Path
from datetime import datetime

class SensorPolicyComparison:
    def __init__(self, data_dir="data", results_dir="results", num_runs=5):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.num_runs = num_runs
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(exist_ok=True)
        
        # Define constants
        self.max_rate = 0.1     # Adjust based on your data
        self.max_delay = 30     # Maximum delay value
        self.M = 1              # Number of sensors to activate per timestep
        self.threshold = 0.5    # Reward threshold (kept for compatibility)
        
        print(f"Data directory: {self.data_dir}")
        print(f"Results directory: {self.results_dir}")
        print(f"Number of runs: {self.num_runs}")

    def load_and_preprocess_data(self, filename="temp_data.csv", max_timesteps=10000):
        """Load and preprocess temperature sensor data"""
        data_path = self.data_dir / filename
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"Loading data from: {data_path}")
        
        # Load dataset and preprocess
        pivot_df = pd.read_csv(data_path)
        pivot_df = pivot_df.apply(lambda x: x.fillna(x.mean()), axis=0)  # Fill missing values
        
        # Restrict to specified number of time steps
        if len(pivot_df) > max_timesteps:
            pivot_df = pivot_df.head(max_timesteps)
            print(f"Restricted to first {max_timesteps} time steps")
        
        print(f"Loaded data shape: {pivot_df.shape}")
        return pivot_df

    def update_node_state_dewma(self, measured_value, last_state_value, last_rate_of_change, delta_t=1, beta_1=0.9, beta_2=0.1):
        """DEWMA update for node state"""
        x1 = beta_1 * measured_value + (1 - beta_1) * (last_state_value + last_rate_of_change * delta_t)
        x2 = beta_2 * ((x1 - last_state_value) / delta_t) + (1 - beta_2) * last_rate_of_change
        return (x1, x2)

    def compute_state(self, rate, delay):
        """Compute continuous state as product of rate and delay"""
        return abs(rate) * delay

    def discretize_state(self, x):
        """Discretize continuous state into bins"""
        bin_edges = np.linspace(0, self.max_rate * self.max_delay, 30)
        return max(1, min(30, np.digitize(x, bin_edges)))

    def passive_transition(self, d):
        """State transition for passive action (not pulled)"""
        return min(d + 1, self.max_delay)

    def active_transition(self, d):
        """State transition for active action (pulled)"""
        # 90% chance to reset delay, 10% chance to increase
        return 1 if random.random() < 0.9 else min(d + 1, self.max_delay)

    def extract_node_id(self, column_name):
        """Extract node ID from column name (e.g., 'node1' -> 1)"""
        try:
            return int(column_name.replace('node', ''))
        except:
            return None

    def run_simulation_RoundRobin(self, pivot_df):
        """Simulate the Round Robin policy"""
        # Remove SN column if present
        if 'SN' in pivot_df.columns:
            pivot_df = pivot_df.drop('SN', axis=1)
        
        N = len(pivot_df.columns)
        T_steps = len(pivot_df)

        # Initialize delays for all sensors (starts at 1)
        delays = np.ones(N, dtype=int)
        
        # Initialize rewards history
        cumulative_reward = 0
        cumulative_rewards = []

        # Track activations by category
        category_counts = {'Category A': 0, 'Category B': 0, 'Category C': 0}

        # Node estimates and sink estimates
        node_estimates = [(20.0, 0.0) for _ in range(N)]
        sink_estimates = [(20.0, 0.0) for _ in range(N)]

        # Round Robin position tracking
        rr_position = 0

        for t in range(1, T_steps):
            # Get current values for all sensors
            current_values = pivot_df.iloc[t].values

            # Update node estimates using DEWMA
            for i in range(N):
                node_estimates[i] = self.update_node_state_dewma(
                    current_values[i],
                    node_estimates[i][0],
                    node_estimates[i][1])

            # Sensor selection using Round Robin
            active_sensors = []
            for j in range(self.M):
                active_sensors.append((rr_position + j) % N)
            
            # Update Round Robin position for next iteration
            rr_position = (rr_position + self.M) % N

            # Convert to action vector
            actions = [1 if i in active_sensors else 0 for i in range(N)]

            # Apply transitions and calculate rewards
            round_reward = 0
            delays_before_transition = delays.copy()

            for i in range(N):
                # State-based reward: NEGATIVE of (delay × |rate|)
                current_state_reward = -(delays_before_transition[i] * abs(sink_estimates[i][1]))
                round_reward += current_state_reward
                
                if actions[i] == 1:  # ACTIVE TRANSITION
                    delays[i] = self.active_transition(delays[i])
                    sink_estimates[i] = node_estimates[i]
                    
                    # Count category activations
                    node_id = self.extract_node_id(pivot_df.columns[i])
                    if node_id is not None:
                        if 1 <= node_id <= 10:
                            category_counts['Category A'] += 1
                        elif 11 <= node_id <= 20:
                            category_counts['Category B'] += 1
                        elif 21 <= node_id <= 30:
                            category_counts['Category C'] += 1
                            
                else:  # PASSIVE TRANSITION
                    delays[i] = self.passive_transition(delays[i])

            # Update cumulative reward
            cumulative_reward += round_reward/N
            cumulative_rewards.append(cumulative_reward / t)

        return cumulative_rewards, category_counts

    def run_simulation_MaxAge(self, pivot_df):
        """Simulate the Maximum Age policy"""
        # Remove SN column if present
        if 'SN' in pivot_df.columns:
            pivot_df = pivot_df.drop('SN', axis=1)
        
        N = len(pivot_df.columns)
        T_steps = len(pivot_df)

        # Initialize delays for all sensors
        delays = np.ones(N, dtype=int)
        
        # Initialize rewards history
        cumulative_reward = 0
        cumulative_rewards = []

        # Track activations by category
        category_counts = {'Category A': 0, 'Category B': 0, 'Category C': 0}

        # Node estimates and sink estimates
        node_estimates = [(20.0, 0.0) for _ in range(N)]
        sink_estimates = [(20.0, 0.0) for _ in range(N)]

        for t in range(1, T_steps):
            # Get current values for all sensors
            current_values = pivot_df.iloc[t].values

            # Update node estimates using DEWMA
            for i in range(N):
                node_estimates[i] = self.update_node_state_dewma(
                    current_values[i],
                    node_estimates[i][0],
                    node_estimates[i][1])

            # Sensor selection based on Maximum Age (highest delays)
            active_sensors = list(np.argsort(delays)[-self.M:])

            # Convert to action vector
            actions = [1 if i in active_sensors else 0 for i in range(N)]

            # Apply transitions and calculate rewards
            round_reward = 0
            delays_before_transition = delays.copy()

            for i in range(N):
                # State-based reward: NEGATIVE of (delay × |rate|)
                current_state_reward = -(delays_before_transition[i] * abs(sink_estimates[i][1]))
                round_reward += current_state_reward
                
                if actions[i] == 1:  # ACTIVE TRANSITION
                    delays[i] = self.active_transition(delays[i])
                    sink_estimates[i] = node_estimates[i]
                    
                    # Count category activations
                    node_id = self.extract_node_id(pivot_df.columns[i])
                    if node_id is not None:
                        if 1 <= node_id <= 10:
                            category_counts['Category A'] += 1
                        elif 11 <= node_id <= 20:
                            category_counts['Category B'] += 1
                        elif 21 <= node_id <= 30:
                            category_counts['Category C'] += 1
                            
                else:  # PASSIVE TRANSITION
                    delays[i] = self.passive_transition(delays[i])

            # Update cumulative reward
            cumulative_reward += round_reward/N
            cumulative_rewards.append(cumulative_reward / t)

        return cumulative_rewards, category_counts

    def run_simulation_WIQL(self, pivot_df):
        """Simulate the WIQL policy"""
        # Remove SN column if present
        if 'SN' in pivot_df.columns:
            pivot_df = pivot_df.drop('SN', axis=1)
        
        N = len(pivot_df.columns)
        T_steps = len(pivot_df)

        # Initialize delays for all sensors
        delays = np.ones(N, dtype=int)
        
        # Initialize rewards history
        cumulative_reward = 0
        cumulative_rewards = []

        # Track activations by category
        category_counts = {'Category A': 0, 'Category B': 0, 'Category C': 0}

        # Node estimates and sink estimates
        node_estimates = [(20.0, 0.0) for _ in range(N)]
        sink_estimates = [(20.0, 0.0) for _ in range(N)]

        # Initialize Q-values and visit counts
        max_state = 30
        Q = [{s: {0: 0.0, 1: 0.0} for s in range(1, max_state+1)} for _ in range(N)]
        counts = [{s: {0: 0, 1: 0} for s in range(1, max_state+1)} for _ in range(N)]

        for t in range(1, T_steps):
            # Get current values for all sensors
            current_values = pivot_df.iloc[t].values

            # Update node estimates using DEWMA
            for i in range(N):
                node_estimates[i] = self.update_node_state_dewma(
                    current_values[i],
                    node_estimates[i][0],
                    node_estimates[i][1])

            # Sensor selection (epsilon-greedy based on Whittle indices)
            eps = N / (N + t)  # Decaying epsilon for exploration

            if random.random() < eps:
                # Random exploration
                active_sensors = random.sample(range(N), self.M)
            else:
                # Exploitation based on Whittle indices
                whittle_indices = np.zeros(N)
                for i in range(N):
                    state_val = self.compute_state(sink_estimates[i][1], delays[i])
                    discrete_state = self.discretize_state(state_val)
                    whittle_indices[i] = Q[i][discrete_state][1] - Q[i][discrete_state][0]

                # Select top M sensors by Whittle index
                active_sensors = list(np.argsort(whittle_indices)[-self.M:])

            # Convert to action vector
            actions = [1 if i in active_sensors else 0 for i in range(N)]

            # Apply transitions and calculate rewards
            round_reward = 0
            delays_before_transition = delays.copy()

            for i in range(N):
                # State-based reward: NEGATIVE of (delay × |rate|)
                current_state_reward = -(delays_before_transition[i] * abs(sink_estimates[i][1]))
                round_reward += current_state_reward
                
                if actions[i] == 1:  # ACTIVE TRANSITION
                    delays[i] = self.active_transition(delays[i])
                    sink_estimates[i] = node_estimates[i]
                    
                    # Count category activations
                    node_id = self.extract_node_id(pivot_df.columns[i])
                    if node_id is not None:
                        if 1 <= node_id <= 10:
                            category_counts['Category A'] += 1
                        elif 11 <= node_id <= 20:
                            category_counts['Category B'] += 1
                        elif 21 <= node_id <= 30:
                            category_counts['Category C'] += 1
                            
                else:  # PASSIVE TRANSITION
                    delays[i] = self.passive_transition(delays[i])

            # Update Q-values for all sensors
            for i in range(N):
                # Current state based on sink rate estimate and delay BEFORE transition
                state_val = self.compute_state(sink_estimates[i][1], delays_before_transition[i])
                curr_state = self.discretize_state(state_val)

                # Action taken
                action = actions[i]

                # Calculate next state after transition
                if action == 1:  # Active
                    next_delay = delays[i]
                    next_sink_rate = node_estimates[i][1]
                else:  # Passive
                    next_delay = delays[i]
                    next_sink_rate = sink_estimates[i][1]
                
                next_state_val = self.compute_state(next_sink_rate, next_delay)
                next_state = self.discretize_state(next_state_val)

                # Update visit count
                counts[i][curr_state][action] += 1
                alpha = 1.0 / counts[i][curr_state][action]

                # Calculate immediate reward for this sensor
                immediate_reward = -(delays_before_transition[i] * abs(sink_estimates[i][1]))

                # Calculate max Q-value for next state
                next_val = max(Q[i][next_state].values()) if next_state in Q[i] else 0.0

                # Update Q-value using Bellman equation
                Q[i][curr_state][action] = (1 - alpha) * Q[i][curr_state][action] + alpha * (immediate_reward + next_val)

            # Update cumulative reward
            cumulative_reward += round_reward/N
            cumulative_rewards.append(cumulative_reward / t)

        return cumulative_rewards, category_counts

    def run_single_experiment(self, pivot_df, run_id):
        """Run all three policies for a single experiment"""
        print(f"  Run {run_id + 1}/{self.num_runs}")
        
        # Set random seed for reproducibility
        np.random.seed(42 + run_id)
        random.seed(42 + run_id)
        
        # Run all three simulations
        cumulative_rewards_wiql, category_counts_wiql = self.run_simulation_WIQL(pivot_df.copy())
        cumulative_rewards_rr, category_counts_rr = self.run_simulation_RoundRobin(pivot_df.copy())
        cumulative_rewards_maxage, category_counts_maxage = self.run_simulation_MaxAge(pivot_df.copy())
        
        return {
            'wiql': {'rewards': cumulative_rewards_wiql, 'categories': category_counts_wiql},
            'round_robin': {'rewards': cumulative_rewards_rr, 'categories': category_counts_rr},
            'max_age': {'rewards': cumulative_rewards_maxage, 'categories': category_counts_maxage}
        }

    def run_all_experiments_and_analyze(self, pivot_df):
        """Run all experiments and calculate statistics"""
        print("Running multi-run experiments...")
        
        # Collect results from all runs
        all_results = {
            'wiql': {'rewards': [], 'categories': []},
            'round_robin': {'rewards': [], 'categories': []},
            'max_age': {'rewards': [], 'categories': []}
        }
        
        # Run multiple experiments
        for run_id in range(self.num_runs):
            single_run_results = self.run_single_experiment(pivot_df, run_id)
            
            for policy in ['wiql', 'round_robin', 'max_age']:
                all_results[policy]['rewards'].append(single_run_results[policy]['rewards'])
                all_results[policy]['categories'].append(single_run_results[policy]['categories'])
        
        # Calculate statistics across runs
        averaged_results = {}
        for policy in ['wiql', 'round_robin', 'max_age']:
            # Calculate mean and std for reward curves
            reward_curves = np.array(all_results[policy]['rewards'])
            mean_rewards = np.mean(reward_curves, axis=0)
            std_rewards = np.std(reward_curves, axis=0)
            
            # Calculate mean and std for category counts
            category_data = all_results[policy]['categories']
            mean_categories = {}
            std_categories = {}
            
            for category in ['Category A', 'Category B', 'Category C']:
                cat_values = [run_data[category] for run_data in category_data]
                mean_categories[category] = np.mean(cat_values)
                std_categories[category] = np.std(cat_values)
            
            averaged_results[policy] = {
                'mean_rewards': mean_rewards,
                'std_rewards': std_rewards,
                'mean_categories': mean_categories,
                'std_categories': std_categories,
                'all_rewards': reward_curves,
                'all_categories': category_data
            }
        
        return averaged_results
        """Create and save performance comparison plots"""
    
        
    def create_performance_plots(self, averaged_results, save_plots=True):
        """Create and save performance comparison plots with confidence intervals"""
        
        # Plot 1: Cumulative Average Rewards with Confidence Intervals
        plt.figure(figsize=(6, 4))
        
        # Sample every 100 points for cleaner visualization
        sample_interval = 100
        time_steps = range(0, len(averaged_results['wiql']['mean_rewards']), sample_interval)
        
        policies = ['wiql', 'round_robin', 'max_age']
        colors = ['orange', 'green', 'blue']
        labels = ['WIQL-UCB', 'RR', 'AoI']
        linestyles = ['-.', '-', '--']
        
        for policy, color, label, style in zip(policies, colors, labels, linestyles):
            mean_curve = averaged_results[policy]['mean_rewards']
            std_curve = averaged_results[policy]['std_rewards']
            
            # Sample for visualization
            mean_sampled = [mean_curve[i] for i in time_steps]
            std_sampled = [std_curve[i] for i in time_steps]
            
            # Plot mean
            plt.plot(time_steps, mean_sampled, label=label, linewidth=2.5, 
                    color=color, linestyle=style)
            
            # Add confidence intervals
            mean_array = np.array(mean_sampled)
            std_array = np.array(std_sampled)
            plt.fill_between(time_steps, 
                           mean_array - std_array, 
                           mean_array + std_array, 
                           alpha=0.2, color=color)
        
        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel("Cumulative Average Reward", fontsize=14)
        #plt.title(f"Sensor Policy Performance Comparison ({self.num_runs} runs)", fontsize=16)
        plt.legend(fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, len(averaged_results['wiql']['mean_rewards']))
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot1_path = self.results_dir / f"sensor_policy_performance_{self.num_runs}runs_{timestamp}.png"
            plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
            print(f"Performance plot saved to: {plot1_path}")
        
        plt.show()

        # Plot 2: Category Activation Distribution with Error Bars
        plt.figure(figsize=(7, 5))
        
        categories = ['Category A', 'Category B', 'Category C']
        
        # Calculate percentages with error bars
        policy_data = {}
        for policy in policies:
            total_mean = sum(averaged_results[policy]['mean_categories'].values())
            percentages = [averaged_results[policy]['mean_categories'][cat] / total_mean * 100 for cat in categories]
            
            # Calculate std for percentages
            std_values = []
            for cat in categories:
                cat_std = averaged_results[policy]['std_categories'][cat]
                cat_mean = averaged_results[policy]['mean_categories'][cat]
                # Approximate std for percentage
                std_pct = (cat_std / total_mean) * 100
                std_values.append(std_pct)
            
            policy_data[policy] = {'percentages': percentages, 'std': std_values}
        
        # Position for grouped bars
        x = np.arange(len(categories))
        width = 0.3
        
        # Plot bars with error bars
        for i, (policy, color, label) in enumerate(zip(policies, colors, labels)):
            offset = (i - 1) * width
            plt.bar(x + offset, policy_data[policy]['percentages'], width, 
                   yerr=policy_data[policy]['std'], capsize=5,
                   label=label, color=color, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
            
            # Add values on top of bars
            for j, (v, std) in enumerate(zip(policy_data[policy]['percentages'], policy_data[policy]['std'])):
                plt.text(j + offset, v + std + 0.5, f"{v:.1f}%", 
                        ha='center', fontsize=10, fontweight='bold')
        
        plt.xlabel("Node Categories", fontsize=14)
        plt.ylabel("Percentage of Polls (%)", fontsize=14)
        #plt.title(f"Category Activation Distribution ({self.num_runs} runs, mean ± std)", fontsize=16)
        plt.legend(fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.xticks(x, categories)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_plots:
            plot2_path = self.results_dir / f"sensor_policy_categories_{self.num_runs}runs_{timestamp}.png"
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
            print(f"Category distribution plot saved to: {plot2_path}")
        
        plt.show()
        
        return plot1_path if save_plots else None, plot2_path if save_plots else None



    def compare_all_policies(self, data_filename="temp_data.csv", max_timesteps=10000):
        """Compare WIQL, Round Robin, and Max Age policies"""
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        print("=" * 60)
        print("SENSOR POLICY COMPARISON")
        print("=" * 60)
        
        # Load and preprocess data
        pivot_df = self.load_and_preprocess_data(data_filename, max_timesteps)
        
        print("Running policy simulations...")
        
        # Run all three simulations
        print("  Running WIQL-UCB simulation...")
        cumulative_rewards_wiql, category_counts_wiql = self.run_simulation_WIQL(pivot_df.copy())
        
        print("  Running Round Robin simulation...")
        cumulative_rewards_rr, category_counts_rr = self.run_simulation_RoundRobin(pivot_df.copy())
        
        print("  Running Max Age (AoI) simulation...")
        cumulative_rewards_maxage, category_counts_maxage = self.run_simulation_MaxAge(pivot_df.copy())
        
        # Organize results
        results = {
            'wiql': {'rewards': cumulative_rewards_wiql, 'categories': category_counts_wiql},
            'round_robin': {'rewards': cumulative_rewards_rr, 'categories': category_counts_rr},
            'max_age': {'rewards': cumulative_rewards_maxage, 'categories': category_counts_maxage}
        }
        
        # Create plots
        self.create_performance_plots(results)
        
        # Save results to CSV
        
        
        # Print detailed results
        print("\n" + "=" * 60)
        print("FINAL PERFORMANCE COMPARISON")
        print("=" * 60)
        print(f"WIQL-UCB Average Reward: {cumulative_rewards_wiql[-1]:.4f}")
        print(f"Round Robin Average Reward: {cumulative_rewards_rr[-1]:.4f}")
        print(f"Max Age (AoI) Average Reward: {cumulative_rewards_maxage[-1]:.4f}")
        
        print("\nCATEGORY ACTIVATION DISTRIBUTION")
        print("-" * 50)
        categories = ['Category A', 'Category B', 'Category C']
        total_activations = sum(category_counts_wiql.values())
        
        wiql_pct = [category_counts_wiql[cat] / total_activations * 100 for cat in categories]
        rr_pct = [category_counts_rr[cat] / total_activations * 100 for cat in categories]
        maxage_pct = [category_counts_maxage[cat] / total_activations * 100 for cat in categories]
        
        print("Category     | WIQL-UCB | Round Robin | Max Age (AoI)")
        print("-" * 50)
        for i, cat in enumerate(categories):
            print(f"{cat.ljust(12)} | {wiql_pct[i]:7.1f}% | {rr_pct[i]:10.1f}% | {maxage_pct[i]:12.1f}%")
        
        print("=" * 60)
        print("COMPARISON COMPLETED!")
        print("=" * 60)
        
    def compare_all_policies(self, data_filename="temp_data.csv", max_timesteps=10000):
        """Compare WIQL, Round Robin, and Max Age policies with multi-run averaging"""
        print("=" * 70)
        print("SENSOR POLICY COMPARISON WITH MULTI-RUN AVERAGING")
        print("=" * 70)
        
        # Load and preprocess data
        pivot_df = self.load_and_preprocess_data(data_filename, max_timesteps)
        
        print(f"Configuration:")
        print(f"  Number of runs: {self.num_runs}")
        print(f"  Sensors per timestep: {self.M}")
        print(f"  Max timesteps: {max_timesteps}")
        print(f"  Policies: WIQL-UCB, Round Robin, Age of Information")
        print("=" * 70)
        
        # Run multi-run experiments and get averaged results
        averaged_results = self.run_all_experiments_and_analyze(pivot_df)
        
        # Create plots with confidence intervals
        print("\nCreating performance plots...")
        self.create_performance_plots(averaged_results)
        
       
        # Print detailed statistical results
        print("\n" + "=" * 70)
        print("FINAL PERFORMANCE COMPARISON (MEAN ± STD)")
        print("=" * 70)
        
        final_window = 1000
        policies = ['wiql', 'round_robin', 'max_age']
        policy_names = ['WIQL-UCB', 'RR', 'AoI']
        
        for policy, name in zip(policies, policy_names):
            # Calculate final performance statistics
            final_rewards = [curve[-final_window:].mean() for curve in averaged_results[policy]['all_rewards']]
            final_mean = np.mean(final_rewards)
            final_std = np.std(final_rewards)
            print(f"{name:18}: {final_mean:.6f} ± {final_std:.6f}")
        
        print(f"\nCATEGORY ACTIVATION DISTRIBUTION (MEAN ± STD)")
        print("-" * 70)
        categories = ['Category A', 'Category B', 'Category C']
        
        print(f"{'Category':<12} | {'WIQL-UCB':<15} | {'Round Robin':<15} | {'Age of Info':<15}")
        print("-" * 70)
        
        for cat in categories:
            wiql_mean = averaged_results['wiql']['mean_categories'][cat]
            wiql_std = averaged_results['wiql']['std_categories'][cat]
            wiql_total = sum(averaged_results['wiql']['mean_categories'].values())
            wiql_pct_mean = (wiql_mean / wiql_total) * 100
            wiql_pct_std = (wiql_std / wiql_total) * 100
            
            rr_mean = averaged_results['round_robin']['mean_categories'][cat]
            rr_std = averaged_results['round_robin']['std_categories'][cat]
            rr_total = sum(averaged_results['round_robin']['mean_categories'].values())
            rr_pct_mean = (rr_mean / rr_total) * 100
            rr_pct_std = (rr_std / rr_total) * 100
            
            maxage_mean = averaged_results['max_age']['mean_categories'][cat]
            maxage_std = averaged_results['max_age']['std_categories'][cat]
            maxage_total = sum(averaged_results['max_age']['mean_categories'].values())
            maxage_pct_mean = (maxage_mean / maxage_total) * 100
            maxage_pct_std = (maxage_std / maxage_total) * 100
            
            print(f"{cat:<12} | {wiql_pct_mean:6.1f}±{wiql_pct_std:4.1f}%   | "
                  f"{rr_pct_mean:6.1f}±{rr_pct_std:4.1f}%   | {maxage_pct_mean:6.1f}±{maxage_pct_std:4.1f}%")
        
        # Performance ranking
        print(f"\nPERFORMANCE RANKING (by final mean reward)")
        print("-" * 40)
        final_performances = []
        for policy, name in zip(policies, policy_names):
            final_rewards = [curve[-final_window:].mean() for curve in averaged_results[policy]['all_rewards']]
            final_mean = np.mean(final_rewards)
            final_performances.append((final_mean, name))
        
        final_performances.sort(reverse=True)
        for i, (performance, name) in enumerate(final_performances, 1):
            print(f"{i}. {name:<18}: {performance:.6f}")
        
        print("=" * 70)
        print("MULTI-RUN COMPARISON COMPLETED!")
        print("=" * 70)
        
        return averaged_results



def main():
    parser = argparse.ArgumentParser(description='Compare sensor scheduling policies with multi-run averaging')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs for averaging (default: 5)')
    parser.add_argument('--data-file', type=str, default='temp_data.csv', help='CSV data filename (default: temp_data.csv)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory (default: data)')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory (default: results)')
    parser.add_argument('--max-timesteps', type=int, default=10000, help='Maximum timesteps to process (default: 10000)')
    parser.add_argument('--sensors-per-step', type=int, default=1, help='Number of sensors to activate per timestep (default: 1)')
    
    args = parser.parse_args()
    
    # Initialize comparison class with multi-run capability
    comparator = SensorPolicyComparison(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        num_runs=args.runs
    )
    
    # Update parameters
    comparator.M = args.sensors_per_step
    
    # Run multi-run comparison
    results = comparator.compare_all_policies(
        data_filename=args.data_file,
        max_timesteps=args.max_timesteps
    )


if __name__ == "__main__":
    main()