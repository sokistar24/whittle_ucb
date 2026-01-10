import time
import csv
import numpy as np
import random
from pathlib import Path


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes/1024:.2f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes/1024**2:.2f} MB"
    return f"{num_bytes/1024**3:.2f} GB"


class ComplexityMixin:
    """
    Adds:
      - runtime profiling for WIQL-BAVT, WIQL-UCB, WIQL-AB, QWIC
      - memory estimates (rough, array-equivalent)
      - sweep runtime vs N

    Requires the host class to implement:
      simulate_WIQL()
      simulate_adaptive_WIQL_UCB()
      simulate_two_timescale()
      simulate_QWIC()
    """

    # ----------------------------
    # Memory estimators (rough)
    # ----------------------------
    def estimate_memory_wiql_bavt(self, dtype=np.float64):
        N = int(self.N)
        S = len(self.states)
        A = len(self.actions)
        item = np.dtype(dtype).itemsize

        q_bytes = N * S * A * item
        c_bytes = N * S * A * item
        lam_bytes = N * S * item
        total = q_bytes + c_bytes + lam_bytes
        return {"Q": q_bytes, "counts": c_bytes, "lambda_est": lam_bytes, "total": total}

    def estimate_memory_wiql_ucb(self, dtype=np.float64):
        N = int(self.N)
        S = len(self.states)
        A = len(self.actions)
        item = np.dtype(dtype).itemsize

        q_bytes = N * S * A * item
        c_bytes = N * S * A * item
        lam_bytes = N * S * item
        total = q_bytes + c_bytes + lam_bytes
        return {"Q": q_bytes, "counts": c_bytes, "lambda_est": lam_bytes, "total": total}

    def estimate_memory_two_timescale(self, dtype=np.float64):
        N = int(self.N)
        S = len(self.states)
        A = len(self.actions)
        item = np.dtype(dtype).itemsize

        q_bytes = N * S * A * S * item
        lam_bytes = S * item
        clocks_bytes = S * A * np.dtype(np.int64).itemsize

        total = q_bytes + lam_bytes + clocks_bytes
        return {"Q_s_a_k": q_bytes, "lambda_est": lam_bytes, "local_clocks": clocks_bytes, "total": total}

    def estimate_memory_qwic_grid(self, lambda_grid_size=10, dtype=np.float64):
        N = int(self.N)
        S = len(self.states)
        A = len(self.actions)
        L = int(lambda_grid_size)
        item = np.dtype(dtype).itemsize

        q_bytes = L * N * S * A * item
        w_bytes = N * S * item
        grid_bytes = L * item

        total = q_bytes + w_bytes + grid_bytes
        return {"Q_grid": q_bytes, "whittle_indices": w_bytes, "lambda_grid": grid_bytes, "total": total}

    # ----------------------------
    # Runtime profiling
    # ----------------------------
    def _profile_runtime_once(self, alg_name: str, seed_offset: int = 0):
        np.random.seed(123 + seed_offset)
        random.seed(123 + seed_offset)

        t0 = time.perf_counter()

        if alg_name == "WIQL-BAVT":
            _ = self.simulate_WIQL()
        elif alg_name == "WIQL-UCB":
            _ = self.simulate_adaptive_WIQL_UCB()
        elif alg_name == "WIQL-AB":
            _ = self.simulate_two_timescale()
        elif alg_name == "QWIC":
            _ = self.simulate_QWIC()
        else:
            raise ValueError(f"Unknown algorithm: {alg_name}")

        t1 = time.perf_counter()
        total_seconds = t1 - t0
        ms_per_step = (total_seconds / float(self.T)) * 1000.0
        return total_seconds, ms_per_step

    def profile_runtime_whittle(self, num_profile_runs=5, dtype=np.float64, qwic_lambda_grid_size=10):
        algs = ["WIQL-BAVT", "WIQL-UCB", "WIQL-AB", "QWIC"]
        results = {a: [] for a in algs}

        # warm-up
        for a in algs:
            try:
                _ = self._profile_runtime_once(a, seed_offset=999)
            except Exception as e:
                print(f"Warm-up skipped for {a}: {e}")

        for r in range(int(num_profile_runs)):
            for a in algs:
                try:
                    _, ms_step = self._profile_runtime_once(a, seed_offset=r)
                    results[a].append(ms_step)
                except Exception as e:
                    print(f"  {a} skipped in run {r+1}/{num_profile_runs}: {e}")

        print("\nRuntime profiling (ms/step) at current N, M, T:")
        for a in algs:
            arr = np.array(results[a], dtype=float)
            if arr.size == 0:
                print(f"  {a:10s}: skipped")
            else:
                print(f"  {a:10s}: {arr.mean():.4f} ± {arr.std():.4f} ms/step  (runs={arr.size})")

        mem_bavt = self.estimate_memory_wiql_bavt(dtype=dtype)
        mem_ucb = self.estimate_memory_wiql_ucb(dtype=dtype)
        mem_ab = self.estimate_memory_two_timescale(dtype=dtype)
        mem_qwic = self.estimate_memory_qwic_grid(lambda_grid_size=qwic_lambda_grid_size, dtype=dtype)

        print("\nMemory estimates (rough):")
        print(f"  WIQL-BAVT total: {_format_bytes(mem_bavt['total'])} "
              f"(Q={_format_bytes(mem_bavt['Q'])}, counts={_format_bytes(mem_bavt['counts'])}, lambda={_format_bytes(mem_bavt['lambda_est'])})")
        print(f"  WIQL-UCB  total: {_format_bytes(mem_ucb['total'])} "
              f"(Q={_format_bytes(mem_ucb['Q'])}, counts={_format_bytes(mem_ucb['counts'])}, lambda={_format_bytes(mem_ucb['lambda_est'])})")
        print(f"  WIQL-AB   total: {_format_bytes(mem_ab['total'])} "
              f"(Q_s_a_k={_format_bytes(mem_ab['Q_s_a_k'])}, lambda={_format_bytes(mem_ab['lambda_est'])}, clocks={_format_bytes(mem_ab['local_clocks'])})")
        print(f"  QWIC      total: {_format_bytes(mem_qwic['total'])} "
              f"(Q_grid={_format_bytes(mem_qwic['Q_grid'])}, indices={_format_bytes(mem_qwic['whittle_indices'])}, grid={_format_bytes(mem_qwic['lambda_grid'])})")

        base = float(mem_ucb["total"])
        if base > 0:
            print("\nMemory overhead ratios (relative to WIQL-UCB):")
            print(f"  WIQL-BAVT / WIQL-UCB: {mem_bavt['total']/base:.3f}x")
            print(f"  WIQL-AB   / WIQL-UCB: {mem_ab['total']/base:.3f}x")
            print(f"  QWIC      / WIQL-UCB: {mem_qwic['total']/base:.3f}x")

        return results

    # ----------------------------
    # Sweep runtime vs N
    # ----------------------------
    def sweep_runtime_vs_N_whittle(
        self,
        N_values,
        M_rule="fraction",
        M_fixed=1,
        frac=0.1,
        T_profile=3000,
        num_profile_runs=3,
        dtype=np.float64,
        qwic_lambda_grid_size=10,
        results_dir="results",
        experiment_name="whittle_complexity_sweep"
    ):
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        original_N, original_M, original_T = int(self.N), int(self.M), int(self.T)
        self.T = int(T_profile)

        algs = ["WIQL-BAVT", "WIQL-UCB", "WIQL-AB", "QWIC"]
        rows = []

        for N in N_values:
            self.N = int(N)
            if M_rule == "fixed":
                self.M = int(M_fixed)
            elif M_rule == "fraction":
                self.M = max(1, int(frac * self.N))
            else:
                raise ValueError("M_rule must be 'fixed' or 'fraction'")

            if self.M > self.N:
                raise ValueError(f"Invalid sweep point: M={self.M} cannot exceed N={self.N}")

            stats = {}
            for a in algs:
                # warm-up per N
                try:
                    _ = self._profile_runtime_once(a, seed_offset=777)
                except Exception as e:
                    stats[a] = (np.nan, np.nan, 0, str(e))
                    continue

                ms_runs = []
                for r in range(int(num_profile_runs)):
                    try:
                        _, ms_step = self._profile_runtime_once(a, seed_offset=1000 + r)
                        ms_runs.append(ms_step)
                    except Exception:
                        pass

                arr = np.array(ms_runs, dtype=float)
                if arr.size == 0:
                    stats[a] = (np.nan, np.nan, 0, "no valid runs")
                else:
                    stats[a] = (arr.mean(), arr.std(), int(arr.size), "")

            mem_bavt = self.estimate_memory_wiql_bavt(dtype=dtype)["total"]
            mem_ucb = self.estimate_memory_wiql_ucb(dtype=dtype)["total"]
            mem_ab = self.estimate_memory_two_timescale(dtype=dtype)["total"]
            mem_qwic = self.estimate_memory_qwic_grid(lambda_grid_size=qwic_lambda_grid_size, dtype=dtype)["total"]

            row = {
                "N": self.N,
                "M": self.M,
                "T": self.T,

                "ms_step_bavt_mean": stats["WIQL-BAVT"][0],
                "ms_step_bavt_std": stats["WIQL-BAVT"][1],
                "runs_bavt": stats["WIQL-BAVT"][2],

                "ms_step_ucb_mean": stats["WIQL-UCB"][0],
                "ms_step_ucb_std": stats["WIQL-UCB"][1],
                "runs_ucb": stats["WIQL-UCB"][2],

                "ms_step_ab_mean": stats["WIQL-AB"][0],
                "ms_step_ab_std": stats["WIQL-AB"][1],
                "runs_ab": stats["WIQL-AB"][2],

                "ms_step_qwic_mean": stats["QWIC"][0],
                "ms_step_qwic_std": stats["QWIC"][1],
                "runs_qwic": stats["QWIC"][2],

                "mem_bavt_bytes": mem_bavt,
                "mem_ucb_bytes": mem_ucb,
                "mem_ab_bytes": mem_ab,
                "mem_qwic_bytes": mem_qwic,

                "mem_ab_over_ucb": (mem_ab / mem_ucb) if mem_ucb > 0 else np.nan,
                "mem_qwic_over_ucb": (mem_qwic / mem_ucb) if mem_ucb > 0 else np.nan,
            }
            rows.append(row)

            print(
                f"Sweep N={self.N}, M={self.M}, T={self.T}: "
                f"BAVT={row['ms_step_bavt_mean']}, "
                f"UCB={row['ms_step_ucb_mean']}, "
                f"AB={row['ms_step_ab_mean']}, "
                f"QWIC={row['ms_step_qwic_mean']}"
            )

        self.N, self.M, self.T = original_N, original_M, original_T

        csv_path = results_path / f"{experiment_name}_runtime_sweep.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved runtime sweep CSV to: {csv_path}")
        return rows


class SimpleRLExperimentRunner(ComplexityMixin):
    def __init__(self, results_dir="results", num_runs=2, experiment_name="Circulant_dynamics"):
        self.results_dir = Path(results_dir)
        self.num_runs = int(num_runs)
        self.experiment_name = str(experiment_name)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running {self.num_runs} experiments...")

        # Environment parameters
        self.states = [0, 1, 2, 3]
        self.actions = [0, 1]  # 0 passive, 1 active
        self.reward_dict = {0: -1, 1: 0, 2: 0, 3: 1}

        # True transition matrices
        self.P0 = np.array([
            [0.5, 0.0, 0.0, 0.5],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5]
        ], dtype=float)

        self.P1 = np.array([
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0, 0.5]
        ], dtype=float)

        # Known optimal Whittle indices for comparison
        self.optimal_index = {0: -0.5, 1: 0.5, 2: 1.0, 3: -1.0}

        # Defaults (override in main)
        self.N = 5
        self.M = 1
        self.T = 10000
        self.gamma = 0.99

    def sample_next_state(self, s, a):
        probs = self.P1[s] if a == 1 else self.P0[s]
        return int(np.random.choice(self.states, p=probs))

    def simulate_optimal(self):
        X = [random.choice(self.states) for _ in range(self.N)]
        cumulative_reward = 0.0
        cumulative_avg = []

        for t in range(self.T):
            priorities = [self.optimal_index[X[i]] for i in range(self.N)]
            active_arms = np.argsort(priorities)[-self.M:]
            A = [1 if i in active_arms else 0 for i in range(self.N)]

            step_reward = 0.0
            X_next = [None] * self.N
            for i in range(self.N):
                s = int(X[i])
                a = int(A[i])
                r = float(self.reward_dict[s])
                step_reward += r
                X_next[i] = self.sample_next_state(s, a)

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / float(t + 1))
            X = X_next

        return np.array(cumulative_avg, dtype=float)

    def simulate_two_timescale(self):
        X = [random.choice(self.states) for _ in range(self.N)]

        # Q[arm][state][action][target_state]
        Q = {
            arm: {
                s: {
                    a: {k: 0.0 for k in self.states}
                    for a in self.actions
                }
                for s in self.states
            }
            for arm in range(self.N)
        }

        lambda_est = {s: 0.0 for s in self.states}
        local_clocks = {s: {a: 0 for a in self.actions} for s in self.states}

        cumulative_reward = 0.0
        cumulative_avg = []
        epsilon = 0.1

        for t in range(1, self.T + 1):
            if random.random() < epsilon:
                active_arms = random.sample(range(self.N), self.M)
            else:
                priorities = [lambda_est[X[i]] for i in range(self.N)]
                active_arms = np.argsort(priorities)[-self.M:]

            A = [1 if i in active_arms else 0 for i in range(self.N)]

            step_reward = 0.0
            X_next = [None] * self.N

            for i in range(self.N):
                s = int(X[i])
                a = int(A[i])
                r = float(self.reward_dict[s])
                step_reward += r

                s_next = self.sample_next_state(s, a)
                X_next[i] = s_next

                local_clocks[s][a] += 1
                alpha = 0.02

                for k in self.states:
                    old_q = Q[i][s][a][k]
                    current_lambda = lambda_est[k]
                    max_q_next = max(Q[i][s_next][v][k] for v in self.actions)
                    td_target = r - current_lambda * a + self.gamma * max_q_next
                    Q[i][s][a][k] = old_q + alpha * (td_target - old_q)

            beta = 0.005
            for k in self.states:
                q_active = np.mean([Q[i][k][1][k] for i in range(self.N)])
                q_passive = np.mean([Q[i][k][0][k] for i in range(self.N)])
                lambda_est[k] += beta * (q_active - q_passive)

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / float(t))
            X = X_next

        return np.array(cumulative_avg, dtype=float)

    def simulate_QWIC(self):
        X = [random.choice(self.states) for _ in range(self.N)]
        lambda_grid = np.linspace(-1.25, 1.25, 10)

        # Q[l_idx][arm][state][action]
        Q = {
            l_idx: {
                arm: {s: {0: 0.0, 1: 0.0} for s in self.states}
                for arm in range(self.N)
            }
            for l_idx in range(len(lambda_grid))
        }

        whittle_indices = {arm: {s: 0.0 for s in self.states} for arm in range(self.N)}

        cumulative_reward = 0.0
        cumulative_avg = []

        def alpha_t(t_):
            return min(2.0 * (t_ ** (-0.5)), 1.0)

        def epsilon_t(t_):
            return t_ ** (-0.5)

        for t in range(1, self.T + 1):
            learning_rate = alpha_t(t)
            exploration_rate = epsilon_t(t)

            current_whittle = []
            for arm in range(self.N):
                s = int(X[arm])
                current_whittle.append((whittle_indices[arm][s], arm))

            current_whittle.sort(reverse=True)
            active_arms = [arm for _, arm in current_whittle[:self.M]]

            if np.random.random() < exploration_rate:
                active_arms = np.random.choice(self.N, self.M, replace=False).tolist()

            A = [1 if i in active_arms else 0 for i in range(self.N)]

            step_reward = 0.0
            X_next = [None] * self.N
            for i in range(self.N):
                s = int(X[i])
                a = int(A[i])
                r = float(self.reward_dict[s])
                step_reward += r

                X_next[i] = self.sample_next_state(s, a)

                current_lambda = float(whittle_indices[i][s])
                lambda_idx = int(np.argmin(np.abs(lambda_grid - current_lambda)))

                old_q = Q[lambda_idx][i][s][a]
                max_q_next = max(Q[lambda_idx][i][X_next[i]][aa] for aa in self.actions)
                td_target = r - current_lambda * a + self.gamma * max_q_next
                Q[lambda_idx][i][s][a] = (1.0 - learning_rate) * old_q + learning_rate * td_target

            # grid search update
            for arm in range(self.N):
                for s in self.states:
                    best_lambda_idx = 0
                    min_diff = float("inf")
                    for l_idx in range(len(lambda_grid)):
                        q_active = Q[l_idx][arm][s][1]
                        q_passive = Q[l_idx][arm][s][0]
                        diff = abs(q_active - q_passive)
                        if diff < min_diff:
                            min_diff = diff
                            best_lambda_idx = l_idx
                    whittle_indices[arm][s] = float(lambda_grid[best_lambda_idx])

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / float(t))
            X = X_next

        return np.array(cumulative_avg, dtype=float)

    def simulate_WIQL(self):
        X = [random.choice(self.states) for _ in range(self.N)]
        Q = [{s: {a: 0.0 for a in self.actions} for s in self.states} for _ in range(self.N)]
        counts = [{s: {a: 0 for a in self.actions} for s in self.states} for _ in range(self.N)]
        lambda_est = [{s: 0.0 for s in self.states} for _ in range(self.N)]

        cumulative_reward = 0.0
        cumulative_avg = []

        for t in range(1, self.T + 1):
            eps = self.N / (self.N + t)
            if random.random() < eps:
                active_arms = random.sample(range(self.N), self.M)
            else:
                priorities = [lambda_est[i][X[i]] for i in range(self.N)]
                active_arms = np.argsort(priorities)[-self.M:]

            A = [1 if i in active_arms else 0 for i in range(self.N)]

            step_reward = 0.0
            X_next = [None] * self.N
            for i in range(self.N):
                s = int(X[i])
                a = int(A[i])
                r = float(self.reward_dict[s])
                step_reward += r

                counts[i][s][a] += 1
                alpha = 1.0 / float(counts[i][s][a])

                s_next = self.sample_next_state(s, a)
                X_next[i] = s_next

                max_q_next = max(Q[i][s_next].values())
                Q[i][s][a] = (1.0 - alpha) * Q[i][s][a] + alpha * (r + self.gamma * max_q_next)

            for i in range(self.N):
                s = int(X[i])
                lambda_est[i][s] = Q[i][s][1] - Q[i][s][0]

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / float(t))
            X = X_next

        return np.array(cumulative_avg, dtype=float)

    def simulate_adaptive_WIQL_UCB(self):
        X = [random.choice(self.states) for _ in range(self.N)]
        Q = np.zeros((self.N, len(self.states), len(self.actions)), dtype=float)
        counts = np.zeros((self.N, len(self.states), len(self.actions)), dtype=float)
        lambda_est = np.zeros((self.N, len(self.states)), dtype=float)

        cumulative_reward = 0.0
        cumulative_avg = []
        c = 2.0  # UCB exploration parameter

        for t in range(1, self.T + 1):
            ucb_values = np.zeros(self.N, dtype=float)

            for i in range(self.N):
                s = int(X[i])
                ucb_action_values = np.zeros(len(self.actions), dtype=float)

                for a in range(len(self.actions)):
                    if counts[i, s, a] > 0:
                        exploration = c * np.sqrt(np.log(t + 1.0) / counts[i, s, a])
                    else:
                        exploration = c * np.sqrt(np.log(t + 1.0))

                    ucb_action_values[a] = Q[i, s, a] + exploration

                ucb_values[i] = ucb_action_values[1] - ucb_action_values[0]

            active_arms = np.argsort(ucb_values)[-self.M:]
            A = [1 if i in active_arms else 0 for i in range(self.N)]

            step_reward = 0.0
            X_next = [None] * self.N
            for i in range(self.N):
                s = int(X[i])
                a = int(A[i])

                r = float(self.reward_dict[s])
                step_reward += r

                counts[i, s, a] += 1.0
                alpha = 1.0 / counts[i, s, a]

                s_next = self.sample_next_state(s, a)
                X_next[i] = s_next

                max_q_next = max(Q[i, s_next, 0], Q[i, s_next, 1])
                Q[i, s, a] = (1.0 - alpha) * Q[i, s, a] + alpha * (r + self.gamma * max_q_next)

                lambda_est[i, s] = Q[i, s, 1] - Q[i, s, 0]

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / float(t))
            X = X_next

        return np.array(cumulative_avg, dtype=float)


if __name__ == "__main__":
    runner = SimpleRLExperimentRunner(results_dir="results", num_runs=2, experiment_name="Circulant_dynamics")

    # set the same N, M, T you used in reward experiments
    runner.N = 50
    runner.M = 5
    runner.T = 10000
    runner.gamma = 0.99

    # profile runtime + memory at this configuration
    runner.profile_runtime_whittle(num_profile_runs=3, dtype=np.float64, qwic_lambda_grid_size=10)

    # optional sweep (runtime scaling)
    runner.sweep_runtime_vs_N_whittle(
        N_values=[10, 30, 50, 100],
        M_rule="fraction",
        frac=0.1,
        T_profile=3000,
        num_profile_runs=3,
        dtype=np.float64,
        qwic_lambda_grid_size=10,
        results_dir="results",
        experiment_name="Circulant_whittle"
    )
