import numpy as np
import math
import random
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


class SimpleRLExperimentRunner:
    """
    Wireless RMAB experiment runner comparing four Whittle-index RL variants:
      1) Two-timescale  -> label: WIQL-AB
      2) QWIC           -> label: WIQL-Fu
      3) WIQL           -> label: WIQL-BAVT
      4) WIQL-UCB       -> label: WIQL-UCB

    Environment (wireless jobs):
      - Each arm is a job with remaining size y (kb) and i.i.d. channel state r each slot.
      - Action a=1 serves the job (reduces y by R_slot), a=0 does not serve.
      - Reward is holding cost: 0 if complete else -c_k.
      - Optional arrivals: when a job completes, it can be replaced with a new job with Poisson rate.
    """

    # =========================================================
    # Wireless RMAB environment
    # =========================================================
    class WirelessRMABEnv:
        def __init__(self,
                     N,
                     M,
                     gamma,
                     seed=123,
                     slot_s=1.67e-3,
                     B=20,
                     mean_job_kb=102.57,
                     arrival_rate=0.0,
                     c_cost=None,
                     R_kbps=None,
                     q=None,
                     arm_class=None):
            self.N = int(N)
            self.M = int(M)
            self.gamma = float(gamma)

            self.actions = [0, 1]  # passive, active

            self.rng = np.random.default_rng(seed)
            random.seed(seed)

            self.slot_s = float(slot_s)

            # discretisation for remaining size y
            self.B = int(B)
            self.y_max_kb = 5.0 * float(mean_job_kb)  # 5 * 102.57 kb
            self.delta_y = self.y_max_kb / self.B
            self.mean_job_kb = float(mean_job_kb)

            # costs
            if c_cost is None:
                c_cost = {1: 1.0, 2: 1.0}
            self.c_cost = dict(c_cost)

            # rates/probabilities per class
            if R_kbps is None:
                R_kbps = {
                    1: [102.6, 204.8, 614.4, 1228.8, 2457.6],
                    2: [102.6, 204.8, 614.4],
                }
            if q is None:
                q = {
                    1: [0.05, 0.23, 0.42, 0.21, 0.09],
                    2: [0.15, 0.33, 0.52],
                }

            self.R_kbps = {k: list(R_kbps[k]) for k in R_kbps}
            self.q = {k: list(q[k]) for k in q}

            # convert kb/s to kb/slot
            self.R_slot = {k: [r * self.slot_s for r in self.R_kbps[k]] for k in self.R_kbps}

            # arm classes
            if arm_class is None:
                arm_class = [1 for _ in range(self.N)]
            if len(arm_class) != self.N:
                raise ValueError("arm_class must have length N.")
            self.arm_class = list(arm_class)

            # unified state space: (y_bin, r_idx_unified)
            self.num_r_max = max(len(self.R_slot[k]) for k in self.R_slot)
            self.num_states = (self.B + 1) * self.num_r_max
            self.states = list(range(self.num_states))

            # arrivals (replacement on completion)
            self.arrival_rate = float(arrival_rate)

            # internal state
            self.Y = None
            self.S = None

        # ------------- mapping -------------
        def encode_state(self, y_bin, r_idx):
            return int(y_bin) * self.num_r_max + int(r_idx)

        def decode_state(self, s_id):
            y_bin = int(s_id) // self.num_r_max
            r_idx = int(s_id) % self.num_r_max
            return y_bin, r_idx

        def size_to_bin(self, y_kb):
            if y_kb <= 0.0:
                return 0
            b = int(math.ceil(y_kb / self.delta_y))
            return min(self.B, max(1, b))

        # ------------- reward -------------
        def reward_from_state(self, s_id, k):
            y_bin, _ = self.decode_state(s_id)
            return 0.0 if y_bin == 0 else -float(self.c_cost[k])

        # ------------- sampling -------------
        def sample_job_size_kb(self):
            return float(self.rng.exponential(scale=self.mean_job_kb))

        def sample_channel_local(self, k):
            probs = self.q[k]
            return int(self.rng.choice(np.arange(len(probs)), p=probs))

        def sample_channel_unified(self, k):
            r_local = self.sample_channel_local(k)
            service_kb = self.R_slot[k][r_local]
            r_unified = r_local
            return r_unified, service_kb

        # ------------- env control -------------
        def reset(self):
            self.Y = [self.sample_job_size_kb() for _ in range(self.N)]
            self.S = [None] * self.N
            for i in range(self.N):
                k = self.arm_class[i]
                r_idx, _ = self.sample_channel_unified(k)
                y_bin = self.size_to_bin(self.Y[i])
                self.S[i] = self.encode_state(y_bin, r_idx)
            return self.Y, self.S

        def step_arm(self, y_kb, k, a):
            r_idx, service_kb = self.sample_channel_unified(k)

            if a == 1:
                y_next = max(0.0, float(y_kb) - float(service_kb))
            else:
                y_next = float(y_kb)

            y_bin_next = self.size_to_bin(y_next)
            s_next = self.encode_state(y_bin_next, r_idx)
            return y_next, s_next

        def maybe_replace_completed_job(self, i):
            if self.Y[i] > 0.0:
                return
            if self.arrival_rate <= 0.0:
                return

            p = 1.0 - math.exp(-self.arrival_rate)  # Poisson prob(>=1)
            if self.rng.random() < p:
                self.Y[i] = self.sample_job_size_kb()
                k = self.arm_class[i]
                r_idx, _ = self.sample_channel_unified(k)
                y_bin = self.size_to_bin(self.Y[i])
                self.S[i] = self.encode_state(y_bin, r_idx)

    # =========================================================
    # Runner init
    # =========================================================
    def __init__(self, results_dir="results", num_runs=5, experiment_name="Wireless_WIQL_Comparison"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.num_runs = int(num_runs)
        self.experiment_name = str(experiment_name)

        print(f"Running {self.num_runs} experiments...")

        # simulation parameters
        self.N = 50
        self.M = 5
        self.T = 20000
        self.gamma = 0.99

        # wireless params
        self.slot_s = 1.67e-3
        self.B = 20
        self.mean_job_kb = 102.57
        self.arrival_rate = 0.0

        self.R_kbps = {
            1: [102.6, 204.8, 614.4, 1228.8, 2457.6],
            2: [102.6, 204.8, 614.4],
        }
        self.q = {
            1: [0.05, 0.23, 0.42, 0.21, 0.09],
            2: [0.15, 0.33, 0.52],
        }
        self.c_cost = {1: 1.0, 2: 1.0}

    def make_env(self, seed):
        # half class-1, half class-2
        arm_class = [1] * (self.N // 2) + [2] * (self.N - self.N // 2)

        env = self.WirelessRMABEnv(
            N=self.N,
            M=self.M,
            gamma=self.gamma,
            seed=seed,
            slot_s=self.slot_s,
            B=self.B,
            mean_job_kb=self.mean_job_kb,
            arrival_rate=self.arrival_rate,
            c_cost=self.c_cost,
            R_kbps=self.R_kbps,
            q=self.q,
            arm_class=arm_class
        )
        return env

    # =========================================================
    # Algorithm 1: WIQL-UCB
    # =========================================================
    def simulate_adaptive_WIQL_UCB(self, env, c=2.0):
        Y, S = env.reset()
        Sdim = env.num_states

        Q = np.zeros((env.N, Sdim, 2), dtype=np.float64)
        counts = np.zeros((env.N, Sdim, 2), dtype=np.float64)

        cumulative_reward = 0.0
        cumulative_avg = []

        for t in range(1, self.T + 1):
            ucb_values = np.zeros(env.N, dtype=np.float64)

            for i in range(env.N):
                s = S[i]
                ucb_action_values = np.zeros(2, dtype=np.float64)

                for a in (0, 1):
                    n = counts[i, s, a]
                    if n > 0:
                        exploration = c * np.sqrt(np.log(t + 1) / n)
                    else:
                        exploration = c * np.sqrt(np.log(t + 1))
                    ucb_action_values[a] = Q[i, s, a] + exploration

                ucb_values[i] = ucb_action_values[1] - ucb_action_values[0]

            active_arms = np.argsort(ucb_values)[-env.M:]
            A = np.zeros(env.N, dtype=np.int8)
            A[active_arms] = 1

            step_reward = 0.0
            for i in range(env.N):
                s = S[i]
                a = int(A[i])
                k = env.arm_class[i]

                r = env.reward_from_state(s, k)
                step_reward += r

                counts[i, s, a] += 1.0
                alpha = 1.0 / counts[i, s, a]

                y_next, s_next = env.step_arm(Y[i], k, a)
                Y[i], S[i] = y_next, s_next
                env.maybe_replace_completed_job(i)

                max_q_next = max(Q[i, s_next, 0], Q[i, s_next, 1])
                Q[i, s, a] = (1 - alpha) * Q[i, s, a] + alpha * (r + self.gamma * max_q_next)

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)

        return np.array(cumulative_avg)

    # =========================================================
    # Algorithm 2: WIQL (BAVT)
    # =========================================================
    def simulate_WIQL(self, env):
        Y, S = env.reset()
        Sdim = env.num_states

        Q = np.zeros((env.N, Sdim, 2), dtype=np.float64)
        counts = np.zeros((env.N, Sdim, 2), dtype=np.float64)
        lambda_est = np.zeros((env.N, Sdim), dtype=np.float64)

        cumulative_reward = 0.0
        cumulative_avg = []

        for t in range(1, self.T + 1):
            eps = env.N / (env.N + t)

            if random.random() < eps:
                active_arms = random.sample(range(env.N), env.M)
            else:
                priorities = np.array([lambda_est[i, S[i]] for i in range(env.N)], dtype=np.float64)
                active_arms = np.argsort(priorities)[-env.M:]

            A = np.zeros(env.N, dtype=np.int8)
            A[active_arms] = 1

            step_reward = 0.0
            S_prev = S.copy()

            for i in range(env.N):
                s = S_prev[i]
                a = int(A[i])
                k = env.arm_class[i]

                r = env.reward_from_state(s, k)
                step_reward += r

                counts[i, s, a] += 1.0
                alpha = 1.0 / counts[i, s, a]

                y_next, s_next = env.step_arm(Y[i], k, a)
                Y[i], S[i] = y_next, s_next
                env.maybe_replace_completed_job(i)

                max_q_next = max(Q[i, s_next, 0], Q[i, s_next, 1])
                Q[i, s, a] = (1 - alpha) * Q[i, s, a] + alpha * (r + self.gamma * max_q_next)

            # update Whittle estimate at visited state (using previous S)
            for i in range(env.N):
                s = S_prev[i]
                lambda_est[i, s] = Q[i, s, 1] - Q[i, s, 0]

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)

        return np.array(cumulative_avg)

    # =========================================================
    # Algorithm 3: QWIC (Fu) grid-search
    # =========================================================
    def simulate_QWIC(self, env, lambda_min=-2.0, lambda_max=2.0, num_grid=21):
        Y, S = env.reset()
        Sdim = env.num_states

        lambda_grid = np.linspace(lambda_min, lambda_max, num_grid)
        Q = np.zeros((num_grid, env.N, Sdim, 2), dtype=np.float64)
        whittle = np.zeros((env.N, Sdim), dtype=np.float64)

        cumulative_reward = 0.0
        cumulative_avg = []

        def alpha_t(t):
            return min(2.0 * (t ** -0.5), 1.0)

        def eps_t(t):
            return t ** -0.5

        for t in range(1, self.T + 1):
            lr = alpha_t(t)
            eps = eps_t(t)

            current = np.array([whittle[i, S[i]] for i in range(env.N)], dtype=np.float64)
            active_arms = np.argsort(current)[-env.M:]

            if np.random.random() < eps:
                active_arms = np.random.choice(env.N, env.M, replace=False).tolist()

            A = np.zeros(env.N, dtype=np.int8)
            A[active_arms] = 1

            step_reward = 0.0
            S_prev = S.copy()

            for i in range(env.N):
                s = S_prev[i]
                a = int(A[i])
                k = env.arm_class[i]

                r = env.reward_from_state(s, k)
                step_reward += r

                lam = whittle[i, s]
                l_idx = int(np.argmin(np.abs(lambda_grid - lam)))

                y_next, s_next = env.step_arm(Y[i], k, a)
                Y[i], S[i] = y_next, s_next
                env.maybe_replace_completed_job(i)

                max_q_next = max(Q[l_idx, i, s_next, 0], Q[l_idx, i, s_next, 1])
                td_target = r - lam * a + self.gamma * max_q_next
                Q[l_idx, i, s, a] = (1 - lr) * Q[l_idx, i, s, a] + lr * td_target

            # grid search for each arm/state
            diffs = np.abs(Q[:, :, :, 1] - Q[:, :, :, 0])  # (L,N,S)
            best = np.argmin(diffs, axis=0)                # (N,S)
            whittle = lambda_grid[best]

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)

        return np.array(cumulative_avg)

    # =========================================================
    # Algorithm 4: Two-timescale (AB-style, lightweight)
    # =========================================================
    def simulate_two_timescale(self, env, epsilon=0.1, alpha0=0.05, beta0=0.01):
        Y, S = env.reset()
        Sdim = env.num_states

        Q = np.zeros((env.N, Sdim, 2), dtype=np.float64)
        visit_sa = np.zeros((env.N, Sdim, 2), dtype=np.float64)

        lambda_est = np.zeros(Sdim, dtype=np.float64)
        visit_s = np.zeros(Sdim, dtype=np.float64)

        cumulative_reward = 0.0
        cumulative_avg = []

        for t in range(1, self.T + 1):
            if random.random() < epsilon:
                active_arms = random.sample(range(env.N), env.M)
            else:
                priorities = np.array([lambda_est[S[i]] for i in range(env.N)], dtype=np.float64)
                active_arms = np.argsort(priorities)[-env.M:]

            A = np.zeros(env.N, dtype=np.int8)
            A[active_arms] = 1

            step_reward = 0.0
            S_prev = S.copy()

            # fast update
            for i in range(env.N):
                s = S_prev[i]
                a = int(A[i])
                k = env.arm_class[i]

                r = env.reward_from_state(s, k)
                step_reward += r

                visit_sa[i, s, a] += 1.0
                alpha = alpha0 / (1.0 + 0.001 * visit_sa[i, s, a])

                y_next, s_next = env.step_arm(Y[i], k, a)
                Y[i], S[i] = y_next, s_next
                env.maybe_replace_completed_job(i)

                max_q_next = max(Q[i, s_next, 0], Q[i, s_next, 1])
                td_target = r - lambda_est[s] * a + self.gamma * max_q_next
                Q[i, s, a] = Q[i, s, a] + alpha * (td_target - Q[i, s, a])

            # slow update
            unique_states = set(S_prev)
            for s in unique_states:
                visit_s[s] += 1.0
                beta = beta0 / (1.0 + 0.001 * visit_s[s])

                idx = [i for i in range(env.N) if S_prev[i] == s]
                if len(idx) == 0:
                    continue

                qdiff = np.mean([Q[i, s, 1] - Q[i, s, 0] for i in idx])
                lambda_est[s] = lambda_est[s] + beta * qdiff

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)

        return np.array(cumulative_avg)

    # =========================================================
    # Experiments
    # =========================================================
    def run_single_experiment(self, run_id):
        print(f"  Run {run_id + 1}/{self.num_runs}")

        seed = 42 + run_id
        np.random.seed(seed)
        random.seed(seed)

        env = self.make_env(seed=seed)

        curves = {}
        curves["two_timescale"] = self.simulate_two_timescale(env)
        curves["qwic"] = self.simulate_QWIC(env)
        curves["wiql"] = self.simulate_WIQL(env)
        curves["adaptive_wiql"] = self.simulate_adaptive_WIQL_UCB(env)

        return curves

    def run(self):
        keys = ["two_timescale", "qwic", "wiql", "adaptive_wiql"]
        curves = {k: [] for k in keys}

        for run_id in range(self.num_runs):
            out = self.run_single_experiment(run_id)
            for k in keys:
                curves[k].append(out[k])

        avg_curves = {}
        std_curves = {}
        for k in keys:
            A = np.vstack(curves[k])
            avg_curves[k] = A.mean(axis=0)
            std_curves[k] = A.std(axis=0)

        return avg_curves, std_curves

    # =========================================================
    # Plotting (your requested style)
    # =========================================================
    def create_performance_plot(self, avg_curves, std_curves):
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

            plt.fill_between(
                weeks,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.2,
                color=color
            )

        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel("Cumulative Average Reward", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim(1, self.T)
        plt.tight_layout()

        filename = f"{self.experiment_name}_N{self.N}_M{self.M}_T{self.T}Time_runs{self.num_runs}.png"
        plot_path = self.results_dir / filename

        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nPlot saved to: {plot_path}")
        return plot_path


if __name__ == "__main__":
    runner = SimpleRLExperimentRunner(
        results_dir="results",
        num_runs=5,
        experiment_name="Wireless_WIQL_FourModels_Styled"
    )

    # Adjust as needed
    runner.N = 50
    runner.M = 5
    runner.T = 10000
    runner.gamma = 0.99

    # Optional: dynamic replacements on completion
    runner.arrival_rate = 0.0  # try 0.001, 0.01, etc.

    avg_curves, std_curves = runner.run()
    runner.create_performance_plot(avg_curves, std_curves)

