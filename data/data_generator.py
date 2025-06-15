#!/usr/bin/env python3
"""
Temperature Sensor Data Generator
Generates synthetic temperature data for multiple sensor nodes with different characteristics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from datetime import datetime

class TemperatureDataGenerator:
    def __init__(self, data_dir="data", results_dir="results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"Data will be saved to: {self.data_dir}")
        print(f"Plots will be saved to: {self.results_dir}")
        
        # Default parameters
        self.time_steps = 50000
        self.num_nodes = 30
        self.z_mean = 20.0  # Mean temperature for all nodes
        self.A_i = 5       # Amplitude for all nodes
        
        # Category parameters
        self.categories = {
            'A': {'nodes': range(1, 11), 'P_i': 500, 'sigma_i': 0.1},  # Category A: nodes 1-10
            'B': {'nodes': range(11, 21), 'P_i': 200, 'sigma_i': 0.2}, # Category B: nodes 11-20
            'C': {'nodes': range(21, 31), 'P_i': 50, 'sigma_i': 0.3}  # Category C: nodes 21-30
        }

    def generate_temperature_data(self):
        """Generate temperature data for all sensor nodes"""
        print(f"Generating temperature data for {self.num_nodes} nodes over {self.time_steps} time steps...")
        
        # Define the nodes (sensors)
        nodes = [f'node{i}' for i in range(1, self.num_nodes + 1)]
        
        # Initialize the data dictionary
        data = {node: np.zeros(self.time_steps) for node in nodes}
        data['SN'] = np.arange(self.time_steps)  # Serial number (time steps)
        
        # Generate temperature data for each node based on categories
        for node in nodes:
            node_id = int(node.replace('node', ''))  # Extract numeric ID
            
            # Determine category and parameters
            category = self.get_node_category(node_id)
            P_i = self.categories[category]['P_i']
            sigma_i = self.categories[category]['sigma_i']
            
            # Generate temperature using the equation:
            # T(t) = z_mean + A_i * sin(2π * t / P_i) + noise
            time_array = np.arange(self.time_steps)
            temperature = (self.z_mean + 
                          self.A_i * np.sin((2 * np.pi * time_array) / P_i) + 
                          np.random.normal(loc=0, scale=sigma_i, size=self.time_steps))
            
            data[node] = temperature
            
            if node_id <= 5:  # Print progress for first few nodes
                print(f"  Generated data for {node} (Category {category}): P_i={P_i}, σ_i={sigma_i}")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df

    def get_node_category(self, node_id):
        """Determine which category a node belongs to"""
        for category, params in self.categories.items():
            if node_id in params['nodes']:
                return category
        return 'Unknown'

    def save_data_to_csv(self, df, filename=None):
        """Save DataFrame to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"temp_state_{timestamp}.csv"
        
        csv_path = self.data_dir / filename
        df.to_csv(csv_path, index=False)
        
        print(f"Temperature data saved to: {csv_path}")
        return csv_path

    def create_visualization(self, df, save_plot=True, show_plot=True):
        """Create visualization of temperature dynamics"""
        plt.figure(figsize=(14, 7))
        
        # Select representative nodes from each category
        sample_nodes = ['node1', 'node5', 'node10', 'node15', 'node20', 'node25', 'node30']
        colors = ['red', 'darkred', 'blue', 'green', 'darkgreen', 'purple', 'orange']
        
        # Plot first 1000 time steps for clarity
        plot_steps = min(1000, self.time_steps)
        
        for i, node in enumerate(sample_nodes):
            if node in df.columns:
                node_id = int(node.replace('node', ''))
                category = self.get_node_category(node_id)
                label = f"{node} (Cat {category})"
                plt.plot(df['SN'][:plot_steps], df[node][:plot_steps], 
                        label=label, color=colors[i % len(colors)], linewidth=1.5)
        
        plt.xlabel('Time Steps', fontsize=14)
        plt.ylabel('Temperature (°C)', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title('Temperature Dynamics of Sensor Nodes', fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=12, ncol=3)
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.results_dir / f"temperature_dynamics_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return plot_path if save_plot else None

    def generate_summary_stats(self, df):
        """Generate and save summary statistics"""
        print("\nGenerating summary statistics...")
        
        # Calculate statistics for each category
        summary_data = []
        
        for category, params in self.categories.items():
            node_cols = [f'node{i}' for i in params['nodes']]
            category_data = df[node_cols]
            
            summary_data.append({
                'Category': category,
                'Nodes': f"node{min(params['nodes'])}-node{max(params['nodes'])}",
                'Period_P_i': params['P_i'],
                'Noise_sigma_i': params['sigma_i'],
                'Mean_Temperature': category_data.mean().mean(),
                'Std_Temperature': category_data.std().mean(),
                'Min_Temperature': category_data.min().min(),
                'Max_Temperature': category_data.max().max()
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary statistics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.data_dir / f"temperature_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Summary statistics saved to: {summary_path}")
        print("\nSummary Statistics:")
        print(summary_df.to_string(index=False))
        
        return summary_df

    def run_generation(self, csv_filename=None, create_plot=True, create_summary=True):
        """Run the complete data generation process"""
        print("=" * 60)
        print("TEMPERATURE SENSOR DATA GENERATION")
        print("=" * 60)
        print(f"Parameters:")
        print(f"  Time steps: {self.time_steps}")
        print(f"  Number of nodes: {self.num_nodes}")
        print(f"  Mean temperature: {self.z_mean}°C")
        print(f"  Amplitude: {self.A_i}°C")
        print("=" * 60)
        
        # Generate temperature data
        df = self.generate_temperature_data()
        
        # Save to CSV
        csv_path = self.save_data_to_csv(df, csv_filename)
        
        # Create visualization
        if create_plot:
            self.create_visualization(df)
        
        # Generate summary statistics
        if create_summary:
            self.generate_summary_stats(df)
        
        print("=" * 60)
        print("DATA GENERATION COMPLETED!")
        print(f"Main CSV file: {csv_path}")
        print("=" * 60)
        
        return df, csv_path


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic temperature sensor data')
    parser.add_argument('--time-steps', type=int, default=50000, help='Number of time steps (default: 50000)')
    parser.add_argument('--nodes', type=int, default=30, help='Number of sensor nodes (default: 30)')
    parser.add_argument('--mean-temp', type=float, default=20.0, help='Mean temperature in °C (default: 20.0)')
    parser.add_argument('--amplitude', type=float, default=5.0, help='Temperature amplitude in °C (default: 10.0)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory (default: data)')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory (default: results)')
    parser.add_argument('--filename', type=str, help='Custom CSV filename (default: auto-generated with timestamp)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    parser.add_argument('--no-summary', action='store_true', help='Skip summary statistics')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TemperatureDataGenerator(
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    
    # Update parameters
    generator.time_steps = args.time_steps
    generator.num_nodes = args.nodes
    generator.z_mean = args.mean_temp
    generator.A_i = args.amplitude
    
    # Update categories if number of nodes changed
    if args.nodes != 30:
        nodes_per_category = args.nodes // 3
        remainder = args.nodes % 3
        
        generator.categories = {
            'A': {'nodes': range(1, nodes_per_category + 1), 'P_i': 500, 'sigma_i': 0.1},
            'B': {'nodes': range(nodes_per_category + 1, 2 * nodes_per_category + 1), 'P_i': 400, 'sigma_i': 0.2},
            'C': {'nodes': range(2 * nodes_per_category + 1, args.nodes + 1), 'P_i': 300, 'sigma_i': 0.3}
        }
    
    # Run data generation
    df, csv_path = generator.run_generation(
        csv_filename=args.filename,
        create_plot=not args.no_plot,
        create_summary=not args.no_summary
    )


if __name__ == "__main__":
    main()