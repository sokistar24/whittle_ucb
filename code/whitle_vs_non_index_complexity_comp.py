# ============================================================
# Complexity-only comparison for 4 algorithms:
#   1) WIQL-UCB
#   2) Joint tabular Q-learning (subset actions)
#   3) Joint DQN (subset actions)
#   4) Joint PPO (subset actions; SB3)
#
# Produces:
#   - printed runtime profiling (ms/step) + PPO training time
#   - printed memory estimates (bytes)
#   - runtime sweep CSV + plot (ms/step vs N)
#
# Notes:
# - Joint Q-learning state space is |S|^N and will be infeasible for large N.
#   The code estimates memory anyway and skips runtime if infeasible.
# - Joint DQN/PPO action space is C(N,M) and is guarded by max_actions.
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
import time
import math
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


class RestartingBanditComplexityRunner:
    def __init__(self, results_dir="results", experiment_name="restart_complexity_4algs"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = str(experiment_name)

        # Environment Setup - Restarting Bandit
        self.states = [0, 1, 2, 3, 4]
        self.actions = [0, 1]  # 0: passive, 1: active
        self.a = 0.9

        # Passive mode: upward drift
        self.P0 = np.array([
            [0.1, 0.9, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.9, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.9, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.9],
            [0.1, 0.0, 0.0, 0.0, 0.9],
        ])

        # Active mode: restart to state 0
        self.P1 = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ])

        # Exact Whittle indices from paper (not used in complexity-only)
        self.optimal_index = {0: -0.9, 1: -0.73, 2: -0.5, 3: -0.26, 4: -0.01}

        # Simulation parameters (set in main)
        self.N = 15
        self.M = 3
        self.T = 3000
        self.gamma = 0.99

        # Guards
        self.max_joint_actions = 200000
        self.max_joint_states_for_runtime = 200000000000000  # for joint tabular Q-learning runtime

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------
    def _format_bytes(self, num_bytes):
        if num_bytes is None or (isinstance(num_bytes, float) and np.isnan(num_bytes)):
            return "n/a"
        num_bytes = float(num_bytes)
        if num_bytes < 1024:
            return f"{num_bytes:.0f} B"
        if num_bytes < 1024**2:
            return f"{num_bytes/1024:.2f} KB"
        if num_bytes < 1024**3:
            return f"{num_bytes/1024**2:.2f} MB"
        return f"{num_bytes/1024**3:.2f} GB"

    # ---------------------------------------------------------
    # Environment
    # ---------------------------------------------------------
    def get_reward(self, state, action):
        if action == 0:
            return float(self.a ** int(state))
        return 0.0

    def sample_next_state(self, s, a):
        if a == 1:
            return 0
        probs = self.P0[int(s)]
        return int(np.random.choice(self.states, p=probs))

    # ---------------------------------------------------------
    # Algorithm 1: WIQL-UCB
    # ---------------------------------------------------------
    def simulate_adaptive_WIQL_UCB(self, c=2.0):
        X = [random.choice(self.states) for _ in range(self.N)]
        Q = np.zeros((self.N, len(self.states), len(self.actions)), dtype=float)
        counts = np.zeros((self.N, len(self.states), len(self.actions)), dtype=float)

        cumulative_reward = 0.0
        cumulative_avg = []

        for t in range(1, self.T + 1):
            ucb_values = np.zeros(self.N, dtype=float)

            for i in range(self.N):
                s = int(X[i])
                ucb_action_values = np.zeros(len(self.actions), dtype=float)

                for a in range(len(self.actions)):
                    if counts[i, s, a] > 0:
                        exploration = c * np.sqrt(np.log(t + 1) / counts[i, s, a])
                    else:
                        exploration = c * np.sqrt(np.log(t + 1))
                    ucb_action_values[a] = Q[i, s, a] + exploration

                ucb_values[i] = ucb_action_values[1] - ucb_action_values[0]

            active_arms = np.argsort(ucb_values)[-self.M:]
            A = np.zeros(self.N, dtype=int)
            A[active_arms] = 1

            step_reward = 0.0
            X_next = [None] * self.N

            for i in range(self.N):
                s = int(X[i])
                a = int(A[i])
                r = self.get_reward(s, a)
                step_reward += r

                counts[i, s, a] += 1.0
                alpha = 1.0 / counts[i, s, a]

                s_next = self.sample_next_state(s, a)
                X_next[i] = s_next

                max_q_next = max(Q[i, s_next, 0], Q[i, s_next, 1])
                Q[i, s, a] = (1 - alpha) * Q[i, s, a] + alpha * (r + self.gamma * max_q_next)

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)
            X = X_next

        return np.array(cumulative_avg, dtype=float)

    # ---------------------------------------------------------
    # Algorithm 2: Joint tabular Q-learning (subset actions)
    # ---------------------------------------------------------
    def simulate_joint_Q_learning(self,
                                 epsilon_start=1.0,
                                 epsilon_min=0.05,
                                 epsilon_decay=0.999,
                                 max_states_runtime=None,
                                 max_actions=None):
        S = len(self.states)
        max_states_runtime = self.max_joint_states_for_runtime if max_states_runtime is None else int(max_states_runtime)
        max_actions = self.max_joint_actions if max_actions is None else int(max_actions)

        num_states = S ** self.N
        num_actions = math.comb(self.N, self.M)

        if num_states > max_states_runtime:
            raise ValueError(f"Joint Q-learning runtime infeasible: |S|^N={S}^{self.N}={num_states} > {max_states_runtime}")
        if num_actions > max_actions:
            raise ValueError(f"Joint Q-learning action space too large: C({self.N},{self.M})={num_actions} > {max_actions}")

        action_list = list(combinations(range(self.N), self.M))

        def encode_joint_state_id(X):
            sid = 0
            for i in range(self.N):
                sid = sid * S + int(X[i])
            return sid

        Q = np.zeros((num_states, num_actions), dtype=np.float32)
        counts = np.zeros((num_states, num_actions), dtype=np.int32)

        X = [random.choice(self.states) for _ in range(self.N)]
        s_id = encode_joint_state_id(X)

        eps = float(epsilon_start)
        cumulative_reward = 0.0
        cumulative_avg = []

        for t in range(1, self.T + 1):
            if np.random.rand() < eps:
                a_idx = np.random.randint(0, num_actions)
            else:
                a_idx = int(np.argmax(Q[s_id]))

            chosen_subset = action_list[a_idx]
            A = np.zeros(self.N, dtype=int)
            A[list(chosen_subset)] = 1

            step_reward = 0.0
            X_next = [None] * self.N
            for i in range(self.N):
                si = int(X[i])
                ai = int(A[i])
                step_reward += self.get_reward(si, ai)
                X_next[i] = self.sample_next_state(si, ai)

            s_next_id = encode_joint_state_id(X_next)

            counts[s_id, a_idx] += 1
            alpha = 1.0 / counts[s_id, a_idx]

            td_target = step_reward + self.gamma * float(np.max(Q[s_next_id]))
            Q[s_id, a_idx] = (1.0 - alpha) * Q[s_id, a_idx] + alpha * td_target

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)

            X = X_next
            s_id = s_next_id
            eps = max(epsilon_min, eps * epsilon_decay)

        return np.array(cumulative_avg, dtype=float)

    # ---------------------------------------------------------
    # Algorithm 3: Joint DQN (subset actions)
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
                           max_actions=None):
        max_actions = self.max_joint_actions if max_actions is None else int(max_actions)

        num_actions = math.comb(self.N, self.M)
        if num_actions > max_actions:
            raise ValueError(f"Joint-action DQN action space too large: C({self.N},{self.M})={num_actions} > {max_actions}")

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except Exception as e:
            raise ImportError("PyTorch is required for DQN. Install with: pip install torch") from e

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_states = len(self.states)
        state_dim = self.N * num_states
        action_dim = num_actions
        action_list = list(combinations(range(self.N), self.M))

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

        q_net = QNet(state_dim, int(hidden_dim), action_dim).to(device)
        target_net = QNet(state_dim, int(hidden_dim), action_dim).to(device)
        target_net.load_state_dict(q_net.state_dict())
        target_net.eval()

        optimiser = optim.Adam(q_net.parameters(), lr=float(lr))
        loss_fn = nn.MSELoss()

        s_buf = np.zeros((int(buffer_size), state_dim), dtype=np.float32)
        a_buf = np.zeros((int(buffer_size),), dtype=np.int64)
        r_buf = np.zeros((int(buffer_size),), dtype=np.float32)
        sn_buf = np.zeros((int(buffer_size), state_dim), dtype=np.float32)
        d_buf = np.zeros((int(buffer_size),), dtype=np.float32)

        buf_ptr = 0
        buf_len = 0

        def encode_joint_state(X):
            v = np.zeros((self.N, num_states), dtype=np.float32)
            for i in range(self.N):
                v[i, int(X[i])] = 1.0
            return v.reshape(-1)

        def push(s, a, r, sn, done):
            nonlocal buf_ptr, buf_len
            s_buf[buf_ptr] = s
            a_buf[buf_ptr] = a
            r_buf[buf_ptr] = r
            sn_buf[buf_ptr] = sn
            d_buf[buf_ptr] = 1.0 if done else 0.0
            buf_ptr = (buf_ptr + 1) % int(buffer_size)
            buf_len = min(buf_len + 1, int(buffer_size))

        def sample_batch():
            idx = np.random.randint(0, buf_len, size=int(batch_size))
            return (
                torch.tensor(s_buf[idx], dtype=torch.float32, device=device),
                torch.tensor(a_buf[idx], dtype=torch.int64, device=device),
                torch.tensor(r_buf[idx], dtype=torch.float32, device=device),
                torch.tensor(sn_buf[idx], dtype=torch.float32, device=device),
                torch.tensor(d_buf[idx], dtype=torch.float32, device=device),
            )

        X = [random.choice(self.states) for _ in range(self.N)]
        s = encode_joint_state(X)

        eps = float(epsilon_start)
        cumulative_reward = 0.0
        cumulative_avg = []

        for t in range(1, self.T + 1):
            if np.random.rand() < eps:
                a_idx = np.random.randint(0, action_dim)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    qs = q_net(s_t)
                    a_idx = int(torch.argmax(qs, dim=1).item())

            chosen_subset = action_list[a_idx]
            A = np.zeros(self.N, dtype=int)
            A[list(chosen_subset)] = 1

            step_reward = 0.0
            X_next = [None] * self.N
            for i in range(self.N):
                si = int(X[i])
                ai = int(A[i])
                step_reward += self.get_reward(si, ai)
                X_next[i] = self.sample_next_state(si, ai)

            sn = encode_joint_state(X_next)
            push(s, a_idx, float(step_reward), sn, False)

            if buf_len >= int(warmup_steps) and (t % int(train_every) == 0):
                s_b, a_b, r_b, sn_b, d_b = sample_batch()
                q_sa = q_net(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    max_next = torch.max(target_net(sn_b), dim=1).values
                    y = r_b + (1.0 - d_b) * float(self.gamma) * max_next

                loss = loss_fn(q_sa, y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            if t % int(target_update_every) == 0:
                target_net.load_state_dict(q_net.state_dict())

            cumulative_reward += step_reward
            cumulative_avg.append(cumulative_reward / t)

            X = X_next
            s = sn
            eps = max(float(epsilon_min), eps * float(epsilon_decay))

        return np.array(cumulative_avg, dtype=float)

    # ---------------------------------------------------------
    # Algorithm 4: Joint PPO (subset actions) via SB3
    # ---------------------------------------------------------
    def simulate_joint_PPO(self,
                           total_timesteps=200_000,
                           n_steps=2048,
                           batch_size=256,
                           learning_rate=3e-4,
                           ent_coef=0.01,
                           clip_range=0.2,
                           gae_lambda=0.95,
                           vf_coef=0.5,
                           max_grad_norm=0.5,
                           net_arch=(128, 128),
                           max_actions=None,
                           deterministic_eval=True,
                           verbose=0):
        max_actions = self.max_joint_actions if max_actions is None else int(max_actions)

        num_actions = math.comb(self.N, self.M)
        if num_actions > max_actions:
            raise ValueError(f"Joint-action PPO action space too large: C({self.N},{self.M})={num_actions} > {max_actions}")

        try:
            import gymnasium as gym
            from gymnasium import spaces
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            import torch as th
        except Exception as e:
            raise ImportError(
                "PPO requires gymnasium + stable-baselines3 + torch.\n"
                "Install with: pip install gymnasium stable-baselines3 torch"
            ) from e

        action_list = list(combinations(range(self.N), self.M))
        num_states = len(self.states)
        obs_dim = self.N * num_states

        runner = self

        class JointSchedulingEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                super().__init__()
                self.action_space = spaces.Discrete(num_actions)
                self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
                self._t = 0
                self.X = None

            def _encode(self, X):
                v = np.zeros((runner.N, num_states), dtype=np.float32)
                for i in range(runner.N):
                    v[i, int(X[i])] = 1.0
                return v.reshape(-1)

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self._t = 0
                self.X = [random.choice(runner.states) for _ in range(runner.N)]
                return self._encode(self.X), {}

            def step(self, action_idx):
                chosen_subset = action_list[int(action_idx)]
                A = np.zeros(runner.N, dtype=int)
                A[list(chosen_subset)] = 1

                step_reward = 0.0
                X_next = [None] * runner.N
                for i in range(runner.N):
                    s = int(self.X[i])
                    a = int(A[i])
                    step_reward += runner.get_reward(s, a)
                    X_next[i] = runner.sample_next_state(s, a)

                self.X = X_next
                self._t += 1

                terminated = (self._t >= runner.T)
                truncated = False
                return self._encode(self.X), float(step_reward), terminated, truncated, {}

        env = DummyVecEnv([lambda: JointSchedulingEnv()])

        policy_kwargs = dict(
            activation_fn=th.nn.Tanh,
            net_arch=dict(pi=list(net_arch), vf=list(net_arch)),
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            ent_coef=float(ent_coef),
            clip_range=float(clip_range),
            gae_lambda=float(gae_lambda),
            vf_coef=float(vf_coef),
            max_grad_norm=float(max_grad_norm),
            gamma=float(self.gamma),
            policy_kwargs=policy_kwargs,
            verbose=int(verbose),
        )

        train_t0 = time.perf_counter()
        model.learn(total_timesteps=int(total_timesteps))
        train_t1 = time.perf_counter()
        train_seconds = train_t1 - train_t0

        # evaluate for exactly T steps
        obs = env.reset()
        cumulative_reward = 0.0
        cumulative_avg = []

        eval_t0 = time.perf_counter()
        for t in range(1, self.T + 1):
            action, _ = model.predict(obs, deterministic=bool(deterministic_eval))
            obs, reward, done, info = env.step(action)

            r = float(reward[0])
            cumulative_reward += r
            cumulative_avg.append(cumulative_reward / t)

            if bool(done[0]):
                obs = env.reset()
        eval_t1 = time.perf_counter()
        eval_seconds = eval_t1 - eval_t0

        return np.array(cumulative_avg, dtype=float), train_seconds, eval_seconds

    # ---------------------------------------------------------
    # Memory estimates (aligned with your earlier style)
    # ---------------------------------------------------------
    def estimate_memory_wiql_ucb(self):
        d = len(self.states)
        A = len(self.actions)
        q_bytes = self.N * d * A * np.dtype(np.float64).itemsize
        c_bytes = self.N * d * A * np.dtype(np.float64).itemsize
        lam_bytes = self.N * d * np.dtype(np.float64).itemsize
        total = q_bytes + c_bytes + lam_bytes
        return {"Q": q_bytes, "counts": c_bytes, "lambda_est": lam_bytes, "total": total}

    def estimate_memory_joint_qlearning(self, dtype_q=np.float32, dtype_counts=np.int32):
        S = len(self.states)
        num_states = S ** self.N
        num_actions = math.comb(self.N, self.M)
        q_bytes = int(num_states) * int(num_actions) * np.dtype(dtype_q).itemsize
        c_bytes = int(num_states) * int(num_actions) * np.dtype(dtype_counts).itemsize
        return {
            "num_states": num_states,
            "num_actions": num_actions,
            "Q": q_bytes,
            "counts": c_bytes,
            "total": q_bytes + c_bytes
        }

    def estimate_memory_joint_dqn(self, hidden_dim=64, buffer_size=30000, dtype_float=np.float32, dtype_action=np.int64):
        S = len(self.states)
        state_dim = self.N * S
        num_actions = math.comb(self.N, self.M)

        H = int(hidden_dim)
        # params: (in->H) + (H->H) + (H->out) with biases
        w_params = (state_dim * H + H) + (H * H + H) + (H * num_actions + num_actions)
        w_bytes_one = w_params * np.dtype(dtype_float).itemsize
        w_bytes = 2 * w_bytes_one  # q + target

        cap = int(buffer_size)
        s_bytes = cap * state_dim * np.dtype(dtype_float).itemsize
        sn_bytes = cap * state_dim * np.dtype(dtype_float).itemsize
        a_bytes = cap * np.dtype(dtype_action).itemsize
        r_bytes = cap * np.dtype(dtype_float).itemsize
        d_bytes = cap * np.dtype(dtype_float).itemsize

        replay_bytes = s_bytes + sn_bytes + a_bytes + r_bytes + d_bytes
        return {
            "state_dim": state_dim,
            "num_actions": num_actions,
            "weights_total": w_bytes,
            "replay_total": replay_bytes,
            "total": w_bytes + replay_bytes
        }

    def estimate_memory_joint_ppo_sb3(self, n_steps=2048, net_arch=(128, 128), dtype_float=np.float32):
        S = len(self.states)
        obs_dim = self.N * S
        num_actions = math.comb(self.N, self.M)

        h1, h2 = int(net_arch[0]), int(net_arch[1])

        pi_params = (obs_dim * h1 + h1) + (h1 * h2 + h2) + (h2 * num_actions + num_actions)
        vf_params = (obs_dim * h1 + h1) + (h1 * h2 + h2) + (h2 * 1 + 1)
        weights_bytes = (pi_params + vf_params) * np.dtype(dtype_float).itemsize

        rollout_bytes = (
            int(n_steps) * obs_dim * np.dtype(dtype_float).itemsize +
            int(n_steps) * 1 * np.dtype(dtype_float).itemsize +
            6 * int(n_steps) * np.dtype(dtype_float).itemsize
        )

        return {
            "obs_dim": obs_dim,
            "num_actions": num_actions,
            "weights_total": weights_bytes,
            "rollout_total": rollout_bytes,
            "total": weights_bytes + rollout_bytes
        }

    # ---------------------------------------------------------
    # Runtime profiling
    # ---------------------------------------------------------
    def _profile_runtime_once(self, alg_name, seed_offset=0,
                              wiql_c=2.0,
                              joint_q_kwargs=None,
                              joint_dqn_kwargs=None,
                              joint_ppo_kwargs=None):
        np.random.seed(123 + seed_offset)
        random.seed(123 + seed_offset)

        joint_q_kwargs = joint_q_kwargs or {}
        joint_dqn_kwargs = joint_dqn_kwargs or {}
        joint_ppo_kwargs = joint_ppo_kwargs or {}

        if alg_name == "WIQL-UCB":
            t0 = time.perf_counter()
            _ = self.simulate_adaptive_WIQL_UCB(c=float(wiql_c))
            t1 = time.perf_counter()
            total = t1 - t0
            return {"total_s": total, "ms_per_step": (total / self.T) * 1000.0}

        if alg_name == "Joint Q-Learning":
            t0 = time.perf_counter()
            _ = self.simulate_joint_Q_learning(**joint_q_kwargs)
            t1 = time.perf_counter()
            total = t1 - t0
            return {"total_s": total, "ms_per_step": (total / self.T) * 1000.0}

        if alg_name == "Joint DQN":
            t0 = time.perf_counter()
            _ = self.simulate_joint_DQN(**joint_dqn_kwargs)
            t1 = time.perf_counter()
            total = t1 - t0
            return {"total_s": total, "ms_per_step": (total / self.T) * 1000.0}

        if alg_name == "Joint PPO":
            # PPO returns train and eval timings separately
            curve, train_s, eval_s = self.simulate_joint_PPO(**joint_ppo_kwargs)
            return {
                "train_s": train_s,
                "eval_s": eval_s,
                "eval_ms_per_step": (eval_s / self.T) * 1000.0,
                "end_to_end_s": train_s + eval_s
            }

        raise ValueError(f"Unknown algorithm: {alg_name}")

    def profile_complexity(self,
                           num_profile_runs=3,
                           wiql_c=2.0,
                           joint_q_kwargs=None,
                           joint_dqn_kwargs=None,
                           joint_ppo_kwargs=None):
        joint_q_kwargs = joint_q_kwargs or {}
        joint_dqn_kwargs = joint_dqn_kwargs or {}
        joint_ppo_kwargs = joint_ppo_kwargs or {}

        algs = ["WIQL-UCB", "Joint Q-Learning", "Joint DQN", "Joint PPO"]

        # warm-up (best effort)
        for a in algs:
            try:
                _ = self._profile_runtime_once(
                    a, seed_offset=999,
                    wiql_c=wiql_c,
                    joint_q_kwargs=joint_q_kwargs,
                    joint_dqn_kwargs=joint_dqn_kwargs,
                    joint_ppo_kwargs=joint_ppo_kwargs
                )
            except Exception:
                pass

        results = {a: [] for a in algs}

        for r in range(num_profile_runs):
            for a in algs:
                try:
                    out = self._profile_runtime_once(
                        a, seed_offset=1000 + r,
                        wiql_c=wiql_c,
                        joint_q_kwargs=joint_q_kwargs,
                        joint_dqn_kwargs=joint_dqn_kwargs,
                        joint_ppo_kwargs=joint_ppo_kwargs
                    )
                    results[a].append(out)
                except Exception as e:
                    print(f"  {a} skipped: {e}")

        print("\nRuntime profiling (same N, M, T used for experiment):")
        for a in algs:
            if len(results[a]) == 0:
                print(f"  {a:15s}: skipped")
                continue

            if a != "Joint PPO":
                ms = np.array([x["ms_per_step"] for x in results[a]], dtype=float)
                print(f"  {a:15s}: {ms.mean():.4f} ± {ms.std():.4f} ms/step  (runs={len(ms)})")
            else:
                train_s = np.array([x["train_s"] for x in results[a]], dtype=float)
                eval_ms = np.array([x["eval_ms_per_step"] for x in results[a]], dtype=float)
                print(f"  {a:15s}: eval {eval_ms.mean():.4f} ± {eval_ms.std():.4f} ms/step (runs={len(eval_ms)})")
                print(f"                   training wall-time {train_s.mean():.2f} ± {train_s.std():.2f} s")

        print("\nMemory estimates (rough):")
        mem_w = self.estimate_memory_wiql_ucb()
        print(f"  WIQL-UCB total: {self._format_bytes(mem_w['total'])} "
              f"(Q={self._format_bytes(mem_w['Q'])}, counts={self._format_bytes(mem_w['counts'])}, lambda={self._format_bytes(mem_w['lambda_est'])})")

        mem_jq = self.estimate_memory_joint_qlearning()
        print(f"  Joint Q-Learning total: {self._format_bytes(mem_jq['total'])} "
              f"(states={mem_jq['num_states']}, actions={mem_jq['num_actions']})")

        hd = int(joint_dqn_kwargs.get("hidden_dim", 64))
        bs = int(joint_dqn_kwargs.get("buffer_size", 30000))
        mem_dqn = self.estimate_memory_joint_dqn(hidden_dim=hd, buffer_size=bs)
        print(f"  Joint DQN total: {self._format_bytes(mem_dqn['total'])} "
              f"(actions={mem_dqn['num_actions']}, state_dim={mem_dqn['state_dim']}, weights={self._format_bytes(mem_dqn['weights_total'])}, replay={self._format_bytes(mem_dqn['replay_total'])})")

        n_steps = int(joint_ppo_kwargs.get("n_steps", 2048))
        net_arch = tuple(joint_ppo_kwargs.get("net_arch", (128, 128)))
        mem_ppo = self.estimate_memory_joint_ppo_sb3(n_steps=n_steps, net_arch=net_arch)
        print(f"  Joint PPO total: {self._format_bytes(mem_ppo['total'])} "
              f"(actions={mem_ppo['num_actions']}, obs_dim={mem_ppo['obs_dim']}, weights={self._format_bytes(mem_ppo['weights_total'])}, rollout={self._format_bytes(mem_ppo['rollout_total'])})")

    # ---------------------------------------------------------
    # Runtime sweep vs N (CSV + plot), similar to your earlier code
    # ---------------------------------------------------------
    def sweep_runtime_vs_N(self, N_values, M_rule="fraction", M_fixed=1, frac=0.1,
                           T_profile=3000,
                           num_profile_runs=2,
                           wiql_c=2.0,
                           joint_q_kwargs=None,
                           joint_dqn_kwargs=None,
                           joint_ppo_kwargs=None):
        joint_q_kwargs = joint_q_kwargs or {}
        joint_dqn_kwargs = joint_dqn_kwargs or {}
        joint_ppo_kwargs = joint_ppo_kwargs or {}

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

            # runtime stats
            algs = ["WIQL-UCB", "Joint Q-Learning", "Joint DQN", "Joint PPO"]
            stats = {}

            for alg in algs:
                ms_runs = []
                ppo_train_runs = []
                ppo_eval_runs = []

                for r in range(num_profile_runs):
                    try:
                        out = self._profile_runtime_once(
                            alg,
                            seed_offset=2000 + r,
                            wiql_c=wiql_c,
                            joint_q_kwargs=joint_q_kwargs,
                            joint_dqn_kwargs=joint_dqn_kwargs,
                            joint_ppo_kwargs=joint_ppo_kwargs
                        )
                        if alg != "Joint PPO":
                            ms_runs.append(out["ms_per_step"])
                        else:
                            ppo_train_runs.append(out["train_s"])
                            ppo_eval_runs.append(out["eval_ms_per_step"])
                    except Exception:
                        continue

                if alg != "Joint PPO":
                    if len(ms_runs) > 0:
                        arr = np.array(ms_runs, dtype=float)
                        stats[alg] = (float(arr.mean()), float(arr.std()), len(arr))
                    else:
                        stats[alg] = (np.nan, np.nan, 0)
                else:
                    if len(ppo_eval_runs) > 0:
                        arr = np.array(ppo_eval_runs, dtype=float)
                        stats[alg] = (float(arr.mean()), float(arr.std()), len(arr))
                    else:
                        stats[alg] = (np.nan, np.nan, 0)

            # memory estimates
            mem_w = self.estimate_memory_wiql_ucb()["total"]
            mem_jq = self.estimate_memory_joint_qlearning()["total"]
            hd = int(joint_dqn_kwargs.get("hidden_dim", 64))
            bs = int(joint_dqn_kwargs.get("buffer_size", 30000))
            mem_dqn = self.estimate_memory_joint_dqn(hidden_dim=hd, buffer_size=bs)["total"]
            n_steps = int(joint_ppo_kwargs.get("n_steps", 2048))
            net_arch = tuple(joint_ppo_kwargs.get("net_arch", (128, 128)))
            mem_ppo = self.estimate_memory_joint_ppo_sb3(n_steps=n_steps, net_arch=net_arch)["total"]

            rows.append({
                "N": self.N,
                "M": self.M,
                "T": self.T,

                "ms_step_wiql_mean": stats["WIQL-UCB"][0],
                "ms_step_wiql_std": stats["WIQL-UCB"][1],
                "runs_wiql": stats["WIQL-UCB"][2],

                "ms_step_jointq_mean": stats["Joint Q-Learning"][0],
                "ms_step_jointq_std": stats["Joint Q-Learning"][1],
                "runs_jointq": stats["Joint Q-Learning"][2],

                "ms_step_dqn_mean": stats["Joint DQN"][0],
                "ms_step_dqn_std": stats["Joint DQN"][1],
                "runs_dqn": stats["Joint DQN"][2],

                "ms_step_ppo_eval_mean": stats["Joint PPO"][0],
                "ms_step_ppo_eval_std": stats["Joint PPO"][1],
                "runs_ppo": stats["Joint PPO"][2],

                "mem_wiql_bytes": mem_w,
                "mem_jointq_bytes": mem_jq,
                "mem_dqn_bytes": mem_dqn,
                "mem_ppo_bytes": mem_ppo,
            })

            print(f"Sweep N={self.N}, M={self.M}, T={self.T}: "
                  f"WIQL={rows[-1]['ms_step_wiql_mean']}, "
                  f"JointQ={rows[-1]['ms_step_jointq_mean']}, "
                  f"DQN={rows[-1]['ms_step_dqn_mean']}, "
                  f"PPO(eval)={rows[-1]['ms_step_ppo_eval_mean']}")

        # restore
        self.N, self.M, self.T = original_N, original_M, original_T

        # save CSV
        csv_path = self.results_dir / f"{self.experiment_name}_runtime_sweep.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        # plot
        Ns = np.array([r["N"] for r in rows], dtype=int)

        plt.figure(figsize=(6.5, 4.2))

        def plot_with_band(y_key_mean, y_key_std, label):
            y = np.array([r[y_key_mean] for r in rows], dtype=float)
            s = np.array([r[y_key_std] for r in rows], dtype=float)

            mask = ~np.isnan(y)
            plt.plot(Ns[mask], y[mask], label=label, linewidth=2.5)
            plt.fill_between(Ns[mask], y[mask] - s[mask], y[mask] + s[mask], alpha=0.2)

        plot_with_band("ms_step_wiql_mean", "ms_step_wiql_std", "WIQL-UCB")
        plot_with_band("ms_step_jointq_mean", "ms_step_jointq_std", "Joint Q-Learning")
        plot_with_band("ms_step_dqn_mean", "ms_step_dqn_std", "Joint DQN")
        plot_with_band("ms_step_ppo_eval_mean", "ms_step_ppo_eval_std", "Joint PPO (eval)")

        plt.xlabel("N", fontsize=14)
        plt.ylabel("Runtime (ms/step)", fontsize=14)
        plt.legend(fontsize=10)
        plt.tick_params(axis="both", labelsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = self.results_dir / f"{self.experiment_name}_runtime_sweep.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"\nSaved runtime CSV to: {csv_path}")
        print(f"Saved runtime plot to: {fig_path}")


if __name__ == "__main__":
    runner = RestartingBanditComplexityRunner(
        results_dir="results",
        experiment_name="Restart_Complexity_WIQL_JointQ_DQN_PPO"
    )

    # Same parameters you used for the experiment
    runner.N = 3
    runner.M = 1
    runner.T = 10
    runner.gamma = 0.99

    # Match the hyperparameters you used in your runs (edit only if you changed them)
    joint_q_kwargs = dict(
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        max_states_runtime=20000000000000000000000,
        max_actions=200000
    )

    joint_dqn_kwargs = dict(
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
        max_actions=200000
    )

    joint_ppo_kwargs = dict(
        total_timesteps=200_000,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        net_arch=(128, 128),
        max_actions=200000,
        deterministic_eval=True,
        verbose=0
    )

    # 1) complexity profiling at fixed N, M, T
    runner.profile_complexity(
        num_profile_runs=3,
        wiql_c=2.0,
        joint_q_kwargs=joint_q_kwargs,
        joint_dqn_kwargs=joint_dqn_kwargs,
        joint_ppo_kwargs=joint_ppo_kwargs
    )

    # 2) optional sweep vs N (same style as your earlier complexity code)
    runner.sweep_runtime_vs_N(
        N_values=[5,10, 15],
        M_rule="fraction",
        frac=0.2,
        T_profile=3000,
        num_profile_runs=2,
        wiql_c=2.0,
        joint_q_kwargs=joint_q_kwargs,
        joint_dqn_kwargs=joint_dqn_kwargs,
        joint_ppo_kwargs=joint_ppo_kwargs
    )
