import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


class SimpleRLExperimentRunner:
    def __init__(self, results_dir="results", num_runs=1, experiment_name="NumericalExample_WIQL_vs_QL"):
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
# Greedy / Myopic baseline (non-learning)
# ---------------------------------------------------------
    def simulate_greedy_myopic(self):
        """
        Greedy baseline: select the M arms with the highest instantaneous reward r(s).
        No learning, no look-ahead.
        """
        X = [random.choice(self.states) for _ in range(self.N)]
        cumulative_reward = 0
        cumulative_avg = []

        for t in range(1, self.T + 1):
            # score = immediate reward of current state
            scores = np.array([self.reward_dict[X[i]] for i in range(self.N)], dtype=float)
            active_arms = np.argsort(scores)[-self.M:]
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
    # Joint-state DQN baseline (small-scale only; M must be 1)
    # ---------------------------------------------------------
    def simulate_joint_DQN(self,
                        hidden_dim=64,
                        buffer_size=30000,
                        batch_size=128,
                        lr=1e-3,
                        epsilon_start=1.0,
                        epsilon_min=0.05,
                        epsilon_decay=0.999,
                        target_update_every=200,
                        train_every=1,
                        warmup_steps=500,
                        max_actions=200000):
        """
        Joint-state DQN baseline for the combinatorial scheduling problem.

        Supports general M by enumerating all subsets of size M:
            action = a subset S ⊂ {1..N}, |S|=M
        Action-space size = C(N,M). This becomes infeasible quickly.

        This implementation is intended ONLY for small (N,M) where C(N,M) <= max_actions.
        """

        # ---------------------------------------------------------
        # Guard: combinatorial action explosion
        # ---------------------------------------------------------
        import math
        from itertools import combinations

        num_actions = math.comb(self.N, self.M)
        if num_actions > max_actions:
            raise ValueError(
                f"Joint-action DQN action space too large: C(N,M)=C({self.N},{self.M})={num_actions} "
                f"> max_actions={max_actions}. Use smaller N/M for DQN baseline."
            )

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except Exception as e:
            raise ImportError("PyTorch is required for DQN. Install with: pip install torch") from e

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------------------------------------------------
        # State encoding: one-hot joint state (x1,...,xN)
        # ---------------------------------------------------------
        num_states = len(self.states)       # 4
        state_dim = self.N * num_states     # N*|S|
        action_dim = num_actions            # C(N,M)

        # Precompute all admissible actions as subsets of indices
        # Each action is a tuple of arm indices to activate
        action_list = list(combinations(range(self.N), self.M))

        # ---------------------------------------------------------
        # Q-network
        # ---------------------------------------------------------
        class QNet(nn.Module):
            def __init__(self, in_dim, hid, out_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hid),
                    nn.ReLU(),
                    nn.Linear(hid, hid),
                    nn.ReLU(),
                    nn.Linear(hid, out_dim),
                )

            def forward(self, x):
                return self.net(x)

        q_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        target_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        target_net.load_state_dict(q_net.state_dict())
        target_net.eval()

        optimiser = optim.Adam(q_net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # ---------------------------------------------------------
        # Replay buffer (numpy arrays)
        # store: s, a_idx, r, s_next, done
        # ---------------------------------------------------------
        s_buf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        a_buf = np.zeros((buffer_size,), dtype=np.int64)
        r_buf = np.zeros((buffer_size,), dtype=np.float32)
        sn_buf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        d_buf = np.zeros((buffer_size,), dtype=np.float32)

        buf_ptr = 0
        buf_len = 0

        def encode_joint_state(X):
            v = np.zeros((self.N, num_states), dtype=np.float32)
            for i in range(self.N):
                v[i, X[i]] = 1.0
            return v.reshape(-1)

        def push(s, a, r, sn, done):
            nonlocal buf_ptr, buf_len
            s_buf[buf_ptr] = s
            a_buf[buf_ptr] = a
            r_buf[buf_ptr] = r
            sn_buf[buf_ptr] = sn
            d_buf[buf_ptr] = 1.0 if done else 0.0
            buf_ptr = (buf_ptr + 1) % buffer_size
            buf_len = min(buf_len + 1, buffer_size)

        def sample_batch():
            idx = np.random.randint(0, buf_len, size=batch_size)
            # use torch.tensor (avoids torch<->numpy bridge issues in some setups)
            return (
                torch.tensor(s_buf[idx], dtype=torch.float32, device=device),
                torch.tensor(a_buf[idx], dtype=torch.int64, device=device),
                torch.tensor(r_buf[idx], dtype=torch.float32, device=device),
                torch.tensor(sn_buf[idx], dtype=torch.float32, device=device),
                torch.tensor(d_buf[idx], dtype=torch.float32, device=device),
            )

        # ---------------------------------------------------------
        # Initialise environment
        # ---------------------------------------------------------
        X = [random.choice(self.states) for _ in range(self.N)]
        s = encode_joint_state(X)

        eps = float(epsilon_start)
        cumulative_reward = 0.0
        cumulative_avg = []

        for t in range(1, self.T + 1):
            # epsilon-greedy over subset-actions
            if np.random.rand() < eps:
                a_idx = np.random.randint(0, action_dim)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    qs = q_net(s_t)  # (1, action_dim)
                    a_idx = int(torch.argmax(qs, dim=1).item())

            chosen_subset = action_list[a_idx]
            A = [0] * self.N
            for i in chosen_subset:
                A[i] = 1

            # Step environment
            step_reward = 0.0
            X_next = [None] * self.N
            for i in range(self.N):
                si = X[i]
                ai = A[i]
                ri = self.reward_dict[si]
                step_reward += ri
                X_next[i] = self.sample_next_state(si, ai)

            sn = encode_joint_state(X_next)
            done = False

            # Store transition
            push(s, a_idx, step_reward, sn, done)

            # Train
            if buf_len >= warmup_steps and (t % train_every == 0):
                s_b, a_b, r_b, sn_b, d_b = sample_batch()

                q_sa = q_net(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    max_next = torch.max(target_net(sn_b), dim=1).values
                    y = r_b + (1.0 - d_b) * self.gamma * max_next

                loss = loss_fn(q_sa, y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            if t % target_update_every == 0:
                target_net.load_state_dict(q_net.state_dict())

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)

            X = X_next
            s = sn
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
        greedy_avg = self.simulate_greedy_myopic()
        wiql_ucb_avg = self.simulate_adaptive_WIQL_UCB(c=2.0)
        qlearn_avg = self.simulate_standard_Q_learning(epsilon=0.1)

        # DQN only for small-scale, and only supports M=1
        dqn_avg = None
        if self.M <= 6  and self.N <= 60:
            dqn_avg = self.simulate_joint_DQN()

        return {
            "Optimal": optimal_avg,
            "Greedy": greedy_avg,
            "WIQL-UCB": wiql_ucb_avg,
            "Q-Learning": qlearn_avg,
            "DQN": dqn_avg,  # may be None
        }


    def run(self):
        # include DQN key; we'll only aggregate it if it exists
        curves = {"Optimal": [], "Greedy": [], "WIQL-UCB": [], "Q-Learning": [], "DQN": []}

        for run_id in range(self.num_runs):
            out = self.run_single_experiment(run_id)
            for k in curves:
                curves[k].append(out[k])

        # Stack + aggregate
        agg = {}
        for k, runs in curves.items():
            # DQN may be None for some/all runs; handle gracefully
            valid = [r for r in runs if r is not None]

            if len(valid) == 0:
                continue  # skip DQN entirely if not run

            A = np.vstack(valid)  # shape: (num_valid_runs, T)
            agg[k] = {"mean": A.mean(axis=0), "std": A.std(axis=0), "num_runs": A.shape[0]}

        return agg


    def save_and_plot(self, agg):
        ts = np.arange(1, self.T + 1)

        plt.figure(figsize=(6, 4))

        # Styling for each algorithm
        styles = {
            "Optimal":     {"color": "black",      "linestyle": "--"},
            "Greedy":      {"color": "tab:green",  "linestyle": "-."},
            "WIQL-UCB":    {"color": "tab:blue",   "linestyle": "-"},
            "Q-Learning":  {"color": "tab:orange", "linestyle": "-"},
            "DQN":         {"color": "tab:red",    "linestyle": "-"},
        }

        # plot in a consistent order (only those present in agg)
        plot_order = ["Optimal", "Greedy", "WIQL-UCB", "Q-Learning", "DQN"]

        for alg in plot_order:
            if alg not in agg:
                continue

            mean_curve = agg[alg]["mean"]
            std_curve = agg[alg]["std"]
            style = styles[alg]

            plt.plot(
                ts,
                mean_curve,
                label=alg,
                linewidth=3.0,
                linestyle=style["linestyle"],
                color=style["color"]
            )

            # Add confidence intervals for learning algorithms only
            # (Greedy + Optimal are deterministic given seeds; but keep rule simple)
            if alg not in ["Optimal", "Greedy"]:
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

        fname = f"{self.experiment_name}_N{self.N}_M{self.M}_T{self.T}_runs{self.num_runs}.png"
        fig_path = self.results_dir / fname
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()

        print("\nFinal cumulative average reward at T:")
        for alg in ["Optimal", "Greedy", "WIQL-UCB", "Q-Learning", "DQN"]:
            if alg not in agg:
                continue
            used_runs = agg[alg].get("num_runs", self.num_runs)
            print(
                f"  {alg:12s}: {agg[alg]['mean'][-1]: .6f} ± {agg[alg]['std'][-1]: .6f} "
                f"(runs used={used_runs})"
            )

        print(f"\nSaved plot to: {fig_path}")





if __name__ == "__main__":
    runner = SimpleRLExperimentRunner(results_dir="results", num_runs=5, experiment_name="NumericalExample_WIQL_vs_QL")

    # You can adjust these if you want
    runner.N = 30
    runner.M = 5
    runner.T = 10000
    runner.gamma = 0.99

    agg = runner.run()
    runner.save_and_plot(agg)
