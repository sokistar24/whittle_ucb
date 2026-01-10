import numpy as np
import random
import time
import csv
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
    # Utilities
    # ---------------------------------------------------------
    def _format_bytes(self, num_bytes):
        if num_bytes < 1024:
            return f"{num_bytes} B"
        if num_bytes < 1024**2:
            return f"{num_bytes/1024:.2f} KB"
        if num_bytes < 1024**3:
            return f"{num_bytes/1024**2:.2f} MB"
        return f"{num_bytes/1024**3:.2f} GB"

    def estimate_memory_standard_qlearning(self):
        d = len(self.states)
        A = len(self.actions)
        q_bytes = self.N * d * A * np.dtype(np.float64).itemsize
        c_bytes = self.N * d * A * np.dtype(np.float64).itemsize
        total = q_bytes + c_bytes
        return {"Q": q_bytes, "counts": c_bytes, "total": total}

    def estimate_memory_wiql_ucb(self):
        d = len(self.states)
        A = len(self.actions)
        q_bytes = self.N * d * A * np.dtype(np.float64).itemsize
        c_bytes = self.N * d * A * np.dtype(np.float64).itemsize
        lam_bytes = self.N * d * np.dtype(np.float64).itemsize
        total = q_bytes + c_bytes + lam_bytes
        return {"Q": q_bytes, "counts": c_bytes, "lambda_est": lam_bytes, "total": total}

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
    # WIQL (optional; not used in default comparison)
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
                Q[i][s][a] = (1 - alpha) * Q[i][s][a] + alpha * (r + max_q_next)

            for i in range(self.N):
                s = X[i]
                lambda_est[i][s] = Q[i][s][1] - Q[i][s][0]

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)
            X = X_next

        return np.array(cumulative_avg)

    # ---------------------------------------------------------
    # WIQL-UCB (adaptive UCB exploration)
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
                Q[i, s, a] = (1 - alpha) * Q[i, s, a] + alpha * (r + self.gamma * max_q_next)

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
    # Experiment runner (reward comparison)
    # ---------------------------------------------------------
    def run_single_experiment(self, run_id):
        print(f"  Run {run_id + 1}/{self.num_runs}")

        np.random.seed(42 + run_id)
        random.seed(42 + run_id)

        optimal_avg = self.simulate_optimal()
        wiql_ucb_avg = self.simulate_adaptive_WIQL_UCB(c=2.0)
        qlearn_avg = self.simulate_standard_Q_learning(epsilon=0.1)

        return {
            "Optimal": optimal_avg,
            "WIQL-UCB": wiql_ucb_avg,
            "Q-Learning": qlearn_avg,
        }

    def run(self):
        curves = {"Optimal": [], "WIQL-UCB": [], "Q-Learning": []}

        for run_id in range(self.num_runs):
            out = self.run_single_experiment(run_id)
            for k in curves:
                curves[k].append(out[k])

        agg = {}
        for k, runs in curves.items():
            A = np.vstack(runs)
            agg[k] = {"mean": A.mean(axis=0), "std": A.std(axis=0)}
        return agg

    def save_and_plot(self, agg):
        ts = np.arange(1, self.T + 1)

        plt.figure(figsize=(6, 4))

        styles = {
            "Optimal":     {"color": "black",     "linestyle": "--"},
            "WIQL-UCB":    {"color": "tab:blue",  "linestyle": "-"},
            "Q-Learning":  {"color": "tab:orange","linestyle": "-"},
        }

        for alg, style in styles.items():
            mean_curve = agg[alg]["mean"]
            std_curve = agg[alg]["std"]

            plt.plot(
                ts, mean_curve,
                label=alg,
                linewidth=3.0,
                linestyle=style["linestyle"],
                color=style["color"]
            )

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
        plt.legend(fontsize=14)
        plt.tick_params(axis="both", labelsize=16)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, self.T)
        plt.tight_layout()

        fname = f"{self.experiment_name}_reward_N{self.N}_M{self.M}_T{self.T}_runs{self.num_runs}.png"
        fig_path = self.results_dir / fname
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()

        print("\nFinal cumulative average reward at T:")
        for alg in ["Optimal", "WIQL-UCB", "Q-Learning"]:
            print(f"  {alg:12s}: {agg[alg]['mean'][-1]: .6f} ± {agg[alg]['std'][-1]: .6f}")

        print(f"\nSaved reward plot to: {fig_path}")

    # ---------------------------------------------------------
    # Complexity profiling (runtime + memory)
    # ---------------------------------------------------------
    def _profile_runtime_once(self, alg_name, seed_offset=0):
        np.random.seed(123 + seed_offset)
        random.seed(123 + seed_offset)

        t0 = time.perf_counter()
        if alg_name == "WIQL-UCB":
            _ = self.simulate_adaptive_WIQL_UCB(c=2.0)
        elif alg_name == "Q-Learning":
            _ = self.simulate_standard_Q_learning(epsilon=0.1)
        else:
            raise ValueError(f"Unknown algorithm: {alg_name}")
        t1 = time.perf_counter()

        total_seconds = t1 - t0
        ms_per_step = (total_seconds / self.T) * 1000.0
        return total_seconds, ms_per_step

    def profile_runtime(self, num_profile_runs=5):
        algs = ["Q-Learning", "WIQL-UCB"]
        results = {a: [] for a in algs}

        # warm-up
        _ = self._profile_runtime_once("Q-Learning", seed_offset=999)
        _ = self._profile_runtime_once("WIQL-UCB", seed_offset=999)

        for r in range(num_profile_runs):
            for a in algs:
                _, ms_step = self._profile_runtime_once(a, seed_offset=r)
                results[a].append(ms_step)

        print("\nRuntime profiling (ms/step):")
        for a in algs:
            arr = np.array(results[a], dtype=float)
            print(f"  {a:10s}: {arr.mean():.4f} ± {arr.std():.4f} ms/step  (runs={num_profile_runs})")

        mem_q = self.estimate_memory_standard_qlearning()
        mem_w = self.estimate_memory_wiql_ucb()

        print("\nMemory estimates (float64 arrays):")
        print(f"  Q-Learning total: {self._format_bytes(mem_q['total'])} "
              f"(Q={self._format_bytes(mem_q['Q'])}, counts={self._format_bytes(mem_q['counts'])})")
        print(f"  WIQL-UCB   total: {self._format_bytes(mem_w['total'])} "
              f"(Q={self._format_bytes(mem_w['Q'])}, counts={self._format_bytes(mem_w['counts'])}, "
              f"lambda={self._format_bytes(mem_w['lambda_est'])})")

        overhead = (mem_w["total"] / mem_q["total"]) if mem_q["total"] > 0 else np.nan
        print(f"\nMemory overhead ratio (WIQL-UCB / Q-Learning): {overhead:.3f}x")

    def sweep_runtime_vs_N(self, N_values, M_rule="fraction", M_fixed=1, frac=0.1,
                           T_profile=5000, num_profile_runs=5):
        """
        Sweeps N and profiles runtime for WIQL-UCB and Q-Learning.
        M_rule:
          - "fixed": M = M_fixed
          - "fraction": M = max(1, int(frac * N))
        Saves: CSV + plot (ms/step vs N) into results/
        """
        original_N, original_M, original_T = self.N, self.M, self.T
        self.T = int(T_profile)

        rows = []

        for N in N_values:
            self.N = int(N)
            if M_rule == "fixed":
                self.M = int(M_fixed)
            elif M_rule == "fraction":
                self.M = max(1, int(frac * self.N))
            else:
                raise ValueError("M_rule must be 'fixed' or 'fraction'")

            algs = ["Q-Learning", "WIQL-UCB"]
            stats = {}

            for a in algs:
                # warm-up per N
                _ = self._profile_runtime_once(a, seed_offset=777)

                ms_runs = []
                for r in range(num_profile_runs):
                    _, ms_step = self._profile_runtime_once(a, seed_offset=1000 + r)
                    ms_runs.append(ms_step)

                arr = np.array(ms_runs, dtype=float)
                stats[a] = (arr.mean(), arr.std())

            mem_q = self.estimate_memory_standard_qlearning()["total"]
            mem_w = self.estimate_memory_wiql_ucb()["total"]

            rows.append({
                "N": self.N,
                "M": self.M,
                "T": self.T,
                "ms_per_step_q_mean": stats["Q-Learning"][0],
                "ms_per_step_q_std": stats["Q-Learning"][1],
                "ms_per_step_ucb_mean": stats["WIQL-UCB"][0],
                "ms_per_step_ucb_std": stats["WIQL-UCB"][1],
                "mem_q_bytes": mem_q,
                "mem_ucb_bytes": mem_w,
                "mem_overhead_ratio": mem_w / mem_q if mem_q > 0 else np.nan
            })

            print(f"Sweep N={self.N}, M={self.M}, T={self.T}: "
                  f"Q={stats['Q-Learning'][0]:.4f} ms/step, "
                  f"UCB={stats['WIQL-UCB'][0]:.4f} ms/step")

        # restore
        self.N, self.M, self.T = original_N, original_M, original_T

        # save CSV
        csv_path = self.results_dir / f"{self.experiment_name}_runtime_sweep.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        # plot runtime vs N (use your preferred styling)
        Ns = np.array([r["N"] for r in rows], dtype=int)
        q_mean = np.array([r["ms_per_step_q_mean"] for r in rows], dtype=float)
        u_mean = np.array([r["ms_per_step_ucb_mean"] for r in rows], dtype=float)
        q_std = np.array([r["ms_per_step_q_std"] for r in rows], dtype=float)
        u_std = np.array([r["ms_per_step_ucb_std"] for r in rows], dtype=float)

        plt.figure(figsize=(6, 4))

        plt.plot(Ns, q_mean, label="Q-Learning", linewidth=3.0)
        plt.fill_between(Ns, q_mean - q_std, q_mean + q_std, alpha=0.2)

        plt.plot(Ns, u_mean, label="WIQL-UCB", linewidth=3.0)
        plt.fill_between(Ns, u_mean - u_std, u_mean + u_std, alpha=0.2)

        plt.xlabel("N", fontsize=16)
        plt.ylabel("Runtime (ms/step)", fontsize=16)
        plt.legend(fontsize=14)
        plt.tick_params(axis="both", labelsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = self.results_dir / f"{self.experiment_name}_runtime_sweep.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"\nSaved runtime CSV to: {csv_path}")
        print(f"Saved runtime plot to: {fig_path}")


if __name__ == "__main__":
    runner = SimpleRLExperimentRunner(results_dir="results", num_runs=5, experiment_name="NumericalExample_WIQL_vs_QL")

    # Reward comparison configuration
    runner.N = 50
    runner.M = 5
    runner.T = 10000
    runner.gamma = 0.99

    # 1) reward curves
    agg = runner.run()
    runner.save_and_plot(agg)

    # 2) runtime + memory profiling at the same N, M, T
    runner.profile_runtime(num_profile_runs=5)

    # 3) optional scaling experiment: runtime vs N (comment out if not needed)
    runner.sweep_runtime_vs_N(
        N_values=[10, 30, 50, 100],
        M_rule="fraction",   # or "fixed"
        frac=0.1,            # used only if M_rule="fraction"
        M_fixed=5,           # used only if M_rule="fixed"
        T_profile=5000,
        num_profile_runs=5
    )
