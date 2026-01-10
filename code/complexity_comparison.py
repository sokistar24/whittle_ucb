import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


class SimpleRLExperimentRunner:
    def __init__(self, results_dir="results", num_runs=5, experiment_name="NumericalExample_WIQL_vs_QL"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.num_runs = int(num_runs)
        self.experiment_name = str(experiment_name)

        print(f"Running {self.num_runs} experiments...")

        # =========================================================
        # Environment parameters (given)
        # =========================================================
        self.states = [0, 1, 2, 3]
        self.actions = [0, 1]  # 0: passive, 1: active
        self.reward_dict = {0: -1, 1: 0, 2: 0, 3: 1}

        self.P0 = np.array([
            [0.5, 0.0, 0.0, 0.5],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5]
        ])

        self.P1 = np.array([
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0, 0.5]
        ])

        # Known optimal Whittle indices (given)
        self.optimal_index = {0: -0.5, 1: 0.5, 2: 1.0, 3: -1.0}

        # =========================================================
        # Simulation parameters (defaults; override in main if needed)
        # =========================================================
        self.N = 5        # number of arms
        self.M = 1        # number of arms activated per time step
        self.T = 10000    # total time steps
        self.gamma = 0.99 # discount factor

    # ---------------------------------------------------------
    # Environment dynamics
    # ---------------------------------------------------------
    def sample_next_state(self, s, a):
        probs = self.P1[s] if a == 1 else self.P0[s]
        return np.random.choice(self.states, p=probs)

    # ---------------------------------------------------------
    # Optimal baseline (uses known Whittle indices provided)
    # ---------------------------------------------------------
    def simulate_optimal(self):
        X = [random.choice(self.states) for _ in range(self.N)]
        cumulative_reward = 0
        cumulative_avg = []

        for t in range(1, self.T + 1):
            priorities = [self.optimal_index[X[i]] for i in range(self.N)]
            active_arms = np.argsort(priorities)[-self.M:]
            A = [1 if i in active_arms else 0 for i in range(self.N)]

            step_reward = 0
            X_next = [None] * self.N
            for i in range(self.N):
                s = X[i]
                a = A[i]
                r = self.reward_dict[s]
                step_reward += r
                X_next[i] = self.sample_next_state(s, a)

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)
            X = X_next

        return np.array(cumulative_avg)

    # ---------------------------------------------------------
    # WIQL (your WIQL-BAVT style)
    # ---------------------------------------------------------
    def simulate_WIQL(self):
        X = [random.choice(self.states) for _ in range(self.N)]
        Q = [{s: {a: 0.0 for a in self.actions} for s in self.states} for _ in range(self.N)]
        counts = [{s: {a: 0 for a in self.actions} for s in self.states} for _ in range(self.N)]
        lambda_est = [{s: 0.0 for s in self.states} for _ in range(self.N)]

        cumulative_reward = 0
        cumulative_avg = []

        for t in range(1, self.T + 1):
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
                s = X[i]
                a = A[i]
                r = self.reward_dict[s]
                step_reward += r

                counts[i][s][a] += 1
                alpha = 1.0 / counts[i][s][a]

                s_next = self.sample_next_state(s, a)
                X_next[i] = s_next

                max_q_next = max(Q[i][s_next].values())
                Q[i][s][a] = (1 - alpha) * Q[i][s][a] + alpha * (r +  max_q_next)

            for i in range(self.N):
                s = X[i]
                lambda_est[i][s] = Q[i][s][1] - Q[i][s][0]

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)
            X = X_next

        return np.array(cumulative_avg)

    # ---------------------------------------------------------
    # WIQL-UCB (your adaptive UCB exploration version)
    # ---------------------------------------------------------
    def simulate_adaptive_WIQL_UCB(self, c=2.0):
        X = [random.choice(self.states) for _ in range(self.N)]
        Q = np.zeros((self.N, len(self.states), len(self.actions)))
        counts = np.zeros((self.N, len(self.states), len(self.actions)))
        lambda_est = np.zeros((self.N, len(self.states)))

        cumulative_reward = 0
        cumulative_avg = []

        for t in range(1, self.T + 1):
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
                s = X[i]
                a = A[i]
                r = self.reward_dict[s]
                step_reward += r

                counts[i, s, a] += 1
                alpha = 1.0 / counts[i, s, a]

                s_next = self.sample_next_state(s, a)
                X_next[i] = s_next

                max_q_next = max(Q[i, s_next, 0], Q[i, s_next, 1])
                Q[i, s, a] = (1 - alpha) * Q[i, s, a] + alpha * (r + max_q_next)

                lambda_est[i, s] = Q[i, s, 1] - Q[i, s, 0]

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)
            X = X_next

        return np.array(cumulative_avg)

    # ---------------------------------------------------------
    # Standard Q-learning baseline (no Whittle index learning/usage)
    # ---------------------------------------------------------
    def simulate_standard_Q_learning(self, epsilon=0.1, epsilon_min=0.01, epsilon_decay=0.9995):
        X = [random.choice(self.states) for _ in range(self.N)]

        Q = np.zeros((self.N, len(self.states), len(self.actions)))
        counts = np.zeros((self.N, len(self.states), len(self.actions)))

        cumulative_reward = 0
        cumulative_avg = []
        eps = float(epsilon)

        for t in range(1, self.T + 1):
            # Enforce constraint with a non-index heuristic:
            # explore: random selection; exploit: pick arms with highest Q(s, active)
            if np.random.rand() < eps:
                active_arms = np.random.choice(np.arange(self.N), size=self.M, replace=False)
            else:
                scores = np.zeros(self.N)
                for i in range(self.N):
                    s = X[i]
                    scores[i] = Q[i, s, 1]
                active_arms = np.argsort(scores)[-self.M:]

            A = [1 if i in active_arms else 0 for i in range(self.N)]

            step_reward = 0
            X_next = [None] * self.N
            for i in range(self.N):
                s = X[i]
                a = A[i]
                r = self.reward_dict[s]
                step_reward += r

                counts[i, s, a] += 1
                alpha = 1.0 / counts[i, s, a]

                s_next = self.sample_next_state(s, a)
                X_next[i] = s_next

                max_q_next = max(Q[i, s_next, 0], Q[i, s_next, 1])
                Q[i, s, a] = (1 - alpha) * Q[i, s, a] + alpha * (r + self.gamma * max_q_next)

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)
            X = X_next

            eps = max(epsilon_min, eps * epsilon_decay)

        return np.array(cumulative_avg)

    # ---------------------------------------------------------
    # Experiment runner
    # ---------------------------------------------------------
    def run_single_experiment(self, run_id):
        print(f"  Run {run_id + 1}/{self.num_runs}")

        # reproducibility
        np.random.seed(42 + run_id)
        random.seed(42 + run_id)

        optimal_avg = self.simulate_optimal()
        #wiql_avg = self.simulate_WIQL()
        wiql_ucb_avg = self.simulate_adaptive_WIQL_UCB(c=2.0)
        qlearn_avg = self.simulate_standard_Q_learning(epsilon=0.1)

        return {
            "Optimal": optimal_avg,
            #"wiql": wiql_avg,
            "WIQL-UCB": wiql_ucb_avg,
            "Q-Learning": qlearn_avg,
        }

    def run(self):
        curves = {"Optimal": [], "WIQL-UCB": [], "Q-Learning": []}

        for run_id in range(self.num_runs):
            out = self.run_single_experiment(run_id)
            for k in curves:
                curves[k].append(out[k])

        # Stack + aggregate
        agg = {}
        for k, runs in curves.items():
            A = np.vstack(runs)  # shape: (num_runs, T)
            agg[k] = {
                "mean": A.mean(axis=0),
                "std": A.std(axis=0),
            }

        return agg

    def save_and_plot(self, agg):
        ts = np.arange(1, self.T + 1)

        plt.figure(figsize=(6, 4))

        # Styling for each algorithm
        styles = {
            "Optimal":     {"color": "black",  "linestyle": "--"},
            "WIQL-UCB":    {"color": "tab:blue", "linestyle": "-"},
            "Q-Learning":  {"color": "tab:orange", "linestyle": "-"},
        }

        for alg, style in styles.items():
            mean_curve = agg[alg]["mean"]
            std_curve = agg[alg]["std"]

            plt.plot(
                ts,
                mean_curve,
                label=alg,
                linewidth=3.0,
                linestyle=style["linestyle"],
                color=style["color"]
            )

            # Add confidence intervals for learning algorithms only
            if alg != "Optimal":
                plt.fill_between(
                    ts,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    alpha=0.2,
                    color=style["color"]
                )

        plt.xlabel("Time Step", fontsize=16)
        plt.ylabel("Cumulative Average Reward", fontsize=16)
        # plt.title(f"N={self.N}, M={self.M}", fontsize=16)  # optional

        plt.legend(fontsize=14)
        plt.tick_params(axis="both", labelsize=16)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, self.T)
        plt.tight_layout()

        fname = (
            f"{self.experiment_name}_N{self.N}_M{self.M}_T{self.T}_"
            f"runs{self.num_runs}.png"
        )
        fig_path = self.results_dir / fname
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()

        print("\nFinal cumulative average reward at T:")
        for alg in ["Optimal", "WIQL-UCB", "Q-Learning"]:
            print(
                f"  {alg:12s}: "
                f"{agg[alg]['mean'][-1]: .6f} ± {agg[alg]['std'][-1]: .6f}"
            )

        print(f"\nSaved plot to: {fig_path}")




if __name__ == "__main__":
    runner = SimpleRLExperimentRunner(results_dir="results", num_runs=10, experiment_name="NumericalExample_WIQL_vs_QL")

    # You can adjust these if you want
    runner.N = 50
    runner.M = 5
    runner.T = 10000
    runner.gamma = 0.99

    agg = runner.run()
    runner.save_and_plot(agg)
