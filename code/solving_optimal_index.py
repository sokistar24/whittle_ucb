#!/usr/bin/env python3
"""
Whittle Index Calculator
Computes optimal Whittle indices for restless multi-armed bandit problems
using value iteration and bisection search
"""

import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime

class WhittleIndexCalculator:
    def __init__(self, results_dir="results", data_dir="data"):
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        
        # Create directories if they don't exist
        self.results_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        print(f"Results will be saved to: {self.results_dir}")
        print(f"Data will be saved to: {self.data_dir}")

    def compute_V(self, lambda_val, P0_matrix, P1_matrix, R_matrix, gamma=0.999):
        """
        Computes the differential value function V for all states given a specific lambda_val.
        Uses Value Iteration with a small discount factor (gamma close to 1) for stability.
        Normalizes V by setting V[0] to 0 to get relative values.

        Args:
            lambda_val (float): The current trial value for lambda.
            P0_matrix (np.ndarray): Transition probabilities for action 0 (passive).
            P1_matrix (np.ndarray): Transition probabilities for action 1 (active).
            R_matrix (np.ndarray): Rewards matrix R[state, action].
            gamma (float): Discount factor for value iteration (close to 1 for average reward).

        Returns:
            np.ndarray: A vector of differential values V for all states.
        """
        num_states = len(R_matrix)
        V = np.zeros(num_states)  # Initialize V to all zeros
        epsilon_V = 1e-7          # Convergence tolerance for V
        max_iter_V = 20000        # Safety limit for value iteration

        for iteration in range(max_iter_V):
            V_prev = V.copy()

            for s in range(num_states):  # For each state 's'
                # Calculate Q-values for action 0 and action 1
                # Action 0 (passive): R(s,0) + gamma * sum(P0(s,s') * V(s'))
                Q0_s = R_matrix[s, 0] + gamma * np.sum(P0_matrix[s, :] * V_prev)

                # Action 1 (active): R(s,1) - lambda_val + gamma * sum(P1(s,s') * V(s'))
                Q1_s = R_matrix[s, 1] - lambda_val + gamma * np.sum(P1_matrix[s, :] * V_prev)

                V[s] = max(Q0_s, Q1_s)

            # Normalize V to get differential values (e.g., by setting V[0] = 0)
            # This prevents values from drifting to infinity for gamma close to 1 in average reward context
            V = V - V[0]

            # Check for convergence
            if np.max(np.abs(V - V_prev)) < epsilon_V:
                break

        return V

    def F_function(self, lambda_val, x_target, P0_matrix, P1_matrix, R_matrix, gamma):
        """
        Calculates the difference between the LHS and RHS of the Whittle index equation.
        F(x, lambda) = (LHS) - (RHS)

        Args:
            lambda_val (float): The current trial value for lambda.
            x_target (int): The specific state for which we are calculating F.
            P0_matrix (np.ndarray): Transition probabilities for action 0.
            P1_matrix (np.ndarray): Transition probabilities for action 1.
            R_matrix (np.ndarray): Rewards matrix R[state, action].
            gamma (float): Discount factor used in compute_V.

        Returns:
            float: The value of F(x_target, lambda_val).
        """
        # 1. Compute V(x0, lambda_val) for all states x0
        V_values = self.compute_V(lambda_val, P0_matrix, P1_matrix, R_matrix, gamma)

        # 2. Calculate LHS and RHS of the Whittle index equation for x_target
        LHS = R_matrix[x_target, 1] - lambda_val + np.sum(P1_matrix[x_target, :] * V_values)
        RHS = R_matrix[x_target, 0] + np.sum(P0_matrix[x_target, :] * V_values)

        return LHS - RHS

    def calculate_whittle_indices(self, R_matrix, P0_matrix, P1_matrix, gamma_val=0.9999, 
                                lambda_range=(-5.0, 5.0), epsilon_lambda=1e-5, max_iter_bisection=200):
        """
        Calculate Whittle indices for all states using bisection search
        
        Args:
            R_matrix (np.ndarray): Rewards matrix R[state, action]
            P0_matrix (np.ndarray): Transition probabilities for action 0 (passive)
            P1_matrix (np.ndarray): Transition probabilities for action 1 (active)
            gamma_val (float): Discount factor for value iteration
            lambda_range (tuple): Range for lambda search (low, high)
            epsilon_lambda (float): Tolerance for lambda convergence
            max_iter_bisection (int): Maximum iterations for bisection search
            
        Returns:
            dict: Dictionary containing results and metadata
        """
        num_states = len(R_matrix)
        optimal_whittle_indices = np.zeros(num_states)
        convergence_info = []
        
        lambda_low_init, lambda_high_init = lambda_range

        print("=" * 60)
        print("WHITTLE INDEX CALCULATION")
        print("=" * 60)
        print(f"Number of states: {num_states}")
        print(f"Gamma (discount factor): {gamma_val}")
        print(f"Lambda search range: [{lambda_low_init}, {lambda_high_init}]")
        print(f"Convergence tolerance: {epsilon_lambda}")
        print("=" * 60)

        for x_target_state in range(num_states):
            print(f"\nSolving for λ({x_target_state})...")

            # Reset bounds for each state
            lambda_low = lambda_low_init
            lambda_high = lambda_high_init
            
            current_lambda_found = None
            converged = False

            for i in range(max_iter_bisection):
                lambda_mid = (lambda_low + lambda_high) / 2.0

                # Evaluate F_function at lambda_mid
                f_val = self.F_function(lambda_mid, x_target_state, P0_matrix, P1_matrix, R_matrix, gamma_val)

                if abs(f_val) < epsilon_lambda:
                    current_lambda_found = lambda_mid
                    converged = True
                    print(f"  ✓ Converged: λ({x_target_state}) = {lambda_mid:.6f} (F = {f_val:.2e}) after {i+1} iterations")
                    break
                elif f_val > 0:
                    # F > 0: lambda_mid is too small, need larger lambda
                    lambda_low = lambda_mid
                else:  # f_val < 0
                    # F < 0: lambda_mid is too large, need smaller lambda
                    lambda_high = lambda_mid

            if not converged:
                current_lambda_found = lambda_mid
                print(f"  ⚠ Did not converge: λ({x_target_state}) ≈ {lambda_mid:.6f} (F = {f_val:.2e}) after {max_iter_bisection} iterations")

            optimal_whittle_indices[x_target_state] = current_lambda_found
            
            # Store convergence information
            convergence_info.append({
                'state': x_target_state,
                'lambda': float(current_lambda_found),
                'final_f_value': float(f_val),
                'iterations': i + 1 if converged else max_iter_bisection,
                'converged': converged
            })

        print("\n" + "=" * 60)
        print("CALCULATION COMPLETE")
        print("=" * 60)
        print("Optimal Whittle Indices:")
        for x in range(num_states):
            print(f"  λ({x}) = {optimal_whittle_indices[x]:8.6f}")
        print("=" * 60)

        # Prepare results dictionary
        results = {
            'whittle_indices': {f'state_{i}': float(optimal_whittle_indices[i]) for i in range(num_states)},
            'whittle_indices_array': optimal_whittle_indices.tolist(),
            'problem_parameters': {
                'num_states': int(num_states),
                'gamma': float(gamma_val),
                'lambda_range': lambda_range,
                'epsilon_lambda': float(epsilon_lambda),
                'max_iter_bisection': int(max_iter_bisection)
            },
            'convergence_info': convergence_info,
            'matrices': {
                'R_matrix': R_matrix.tolist(),
                'P0_matrix': P0_matrix.tolist(),
                'P1_matrix': P1_matrix.tolist()
            }
        }

        return results

    def save_results(self, results):
        """Save results to JSON and CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results to JSON
        json_path = self.data_dir / f"whittle_indices_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save indices to simple CSV
        import pandas as pd
        
        # Create simple indices table
        indices_data = {
            'state': list(range(len(results['whittle_indices_array']))),
            'whittle_index': results['whittle_indices_array']
        }
        
        indices_df = pd.DataFrame(indices_data)
        csv_path = self.data_dir / f"whittle_indices_{timestamp}.csv"
        indices_df.to_csv(csv_path, index=False)
        
        # Save convergence info
        convergence_df = pd.DataFrame(results['convergence_info'])
        conv_path = self.data_dir / f"whittle_convergence_{timestamp}.csv"
        convergence_df.to_csv(conv_path, index=False)
        
        print(f"\nResults saved:")
        print(f"  Complete results: {json_path}")
        print(f"  Indices table: {csv_path}")
        print(f"  Convergence info: {conv_path}")
        
        return json_path, csv_path, conv_path

    def run_default_problem(self):
        """Run the default problem from the original code"""
        print("Running default problem setup...")
        
        # Default problem parameters
        R_matrix = np.array([
            [-1, -1],  # State 0: R(0,0)=-1, R(0,1)=-1
            [ 0,  0],  # State 1: R(1,0)=0, R(1,1)=0
            [ 0,  0],  # State 2: R(2,0)=0, R(2,1)=0
            [ 1,  1]   # State 3: R(3,0)=1, R(3,1)=1
        ])

        P1 = np.array([  # Transition matrix for action 1 (active)
            [0.5, 0.5, 0, 0],
            [0, 0.5, 0.5, 0],
            [0, 0, 0.5, 0.5],
            [0.5, 0, 0, 0.5]
        ])

        P0 = np.array([  # Transition matrix for action 0 (passive)
            [0.5, 0, 0, 0.5],
            [0.5, 0.5, 0, 0],
            [0, 0.5, 0.5, 0],
            [0, 0, 0.5, 0.5]
        ])

        # Calculate indices
        results = self.calculate_whittle_indices(R_matrix, P0, P1)
        
        # Save results
        self.save_results(results)
        
        return results

    def run_custom_problem(self, R_matrix, P0_matrix, P1_matrix, gamma_val=0.9999):
        """Run with custom problem matrices"""
        print("Running custom problem setup...")
        
        # Validate inputs
        assert R_matrix.shape[0] == P0_matrix.shape[0] == P1_matrix.shape[0], "Matrix dimensions must match"
        assert R_matrix.shape[1] == 2, "R_matrix must have 2 actions (columns)"
        assert P0_matrix.shape[0] == P0_matrix.shape[1], "P0_matrix must be square"
        assert P1_matrix.shape[0] == P1_matrix.shape[1], "P1_matrix must be square"
        
        # Check that transition matrices are stochastic
        assert np.allclose(P0_matrix.sum(axis=1), 1), "P0_matrix rows must sum to 1"
        assert np.allclose(P1_matrix.sum(axis=1), 1), "P1_matrix rows must sum to 1"
        
        print(f"Custom problem validated: {R_matrix.shape[0]} states, 2 actions")
        
        # Calculate indices
        results = self.calculate_whittle_indices(R_matrix, P0_matrix, P1_matrix, gamma_val)
        
        # Save results
        self.save_results(results)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Calculate Whittle indices for restless multi-armed bandit problems')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory (default: results)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory (default: data)')
    parser.add_argument('--gamma', type=float, default=0.9999, help='Discount factor (default: 0.9999)')
    parser.add_argument('--lambda-low', type=float, default=-5.0, help='Lower bound for lambda search (default: -5.0)')
    parser.add_argument('--lambda-high', type=float, default=5.0, help='Upper bound for lambda search (default: 5.0)')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='Convergence tolerance (default: 1e-5)')
    parser.add_argument('--max-iter', type=int, default=200, help='Maximum bisection iterations (default: 200)')
    parser.add_argument('--custom-matrices', type=str, help='Path to JSON file with custom R, P0, P1 matrices')
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = WhittleIndexCalculator(
        results_dir=args.results_dir,
        data_dir=args.data_dir
    )
    
    if args.custom_matrices:
        # Load custom problem from JSON file
        print(f"Loading custom problem from: {args.custom_matrices}")
        with open(args.custom_matrices, 'r') as f:
            matrices = json.load(f)
        
        R_matrix = np.array(matrices['R_matrix'])
        P0_matrix = np.array(matrices['P0_matrix'])
        P1_matrix = np.array(matrices['P1_matrix'])
        
        results = calculator.run_custom_problem(R_matrix, P0_matrix, P1_matrix, args.gamma)
    else:
        # Run default problem
        results = calculator.run_default_problem()
    
    print("\nWhittle Index Calculation Completed Successfully!")


if __name__ == "__main__":
    main()