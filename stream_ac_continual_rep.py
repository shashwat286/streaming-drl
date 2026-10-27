"""
Stream AC(λ) with Representation-Preserving Continual Learning (RPCL).

Tasks are trained sequentially (default: Humanoid-v4 for 20M steps, then Hopper-v4 for 20M steps)
with ONE model:
    * task-specific input layer   (obs_dim_k -> hidden)    one per task
    * shared trunk                (hidden    -> hidden)    shared by all tasks; output z = f_theta(x)
    * task-specific output heads  (hidden    -> action / value)

After training on task A, a small memory of task-A inputs x_A is stored together with their
representations z_A = f_{theta_A}(x_A) (penultimate features of the actor and of the critic).
While training on task B the model is also pushed to keep f_theta(x_A) close to z_A:

    L(theta) = L_B(theta) + lambda * E_{x_A} [ || f_theta(x_A) - z_A ||_2^2 ]

Stream AC has no explicit loss (it uses ObGD with eligibility traces), so the penalty gradient is
folded into the same ObGD update direction:

    e   <- gamma * lamda * e + grad(objective)              (eligibility trace, as in ObGD)
    d   <- delta * e - lambda * grad(L_rep)                 (combined update direction)
    m   <- max(beta * m, |d|)                               (ObGD's step-size bound, now on d)
    p   <- p + lr * d / (m + eps)

With lambda = 0 this reduces exactly to the original StreamingOptimizer.
"""
import argparse
import os
import pickle
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Normal
from optimizer import StreamingOptimizer
from obs_reward_transforms import NormalizeObservation, ScaleReward
from sparse_init import sparse_init


# ----------------------------------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------------------------------
class RepPreservingStreamingOptimizer(StreamingOptimizer):
    """ObGD whose update direction also contains a (non-traced) auxiliary descent term."""

    @torch.no_grad()
    def step(self, delta=1.0, reset=False, aux_grads=None):
        delta = float(delta)
        for group in self.param_groups:
            trace_decay = group["gamma"] * group["lamda"]
            lr = group["lr"]
            eps = group["eps"]
            for p in group["params"]:
                grad = p.grad
                aux = None if aux_grads is None else aux_grads.get(p)
                if grad is None and aux is None:
                    continue
                state = self.state[p]
                if not state:
                    state["eligibility_trace"] = torch.zeros_like(p)
                    state["max_v"] = torch.zeros_like(p)
                e = state["eligibility_trace"]
                max_v = state["max_v"]
                e.mul_(trace_decay)
                if grad is not None:
                    e.add_(grad)
                direction = e * delta
                if aux is not None:
                    direction.sub_(aux)
                torch.maximum(max_v.mul_(self.beta), direction.abs(), out=max_v)
                p.addcdiv_(direction, max_v + eps, value=lr)
                if reset:
                    e.zero_()


# ----------------------------------------------------------------------------------------------
# Networks (same layers as stream_ac_continuous.py, split into input / trunk / head)
# ----------------------------------------------------------------------------------------------
class InputLayer(nn.Module):
    def __init__(self, n_obs, hidden_size):
        super().__init__()
        self.fc_layer = nn.Linear(n_obs, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, x):
        return F.leaky_relu(self.layernorm1(self.fc_layer(x)))


class Trunk(nn.Module):
    """Shared part. Its output is the representation z = f_theta(x) that is preserved."""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, h):
        return F.leaky_relu(self.layernorm2(self.hidden_layer(h)))


class GaussianHead(nn.Module):
    def __init__(self, hidden_size, n_actions):
        super().__init__()
        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)

    def forward(self, z):
        return self.linear_mu(z), F.softplus(self.linear_std(z))


class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear_layer = nn.Linear(hidden_size, 1)

    def forward(self, z):
        return self.linear_layer(z)


class RepMemory:
    """Stored (x_A, z_A) pairs for one previous task."""
    def __init__(self, task, x, z_actor, z_critic):
        self.task, self.x, self.z_actor, self.z_critic = task, x, z_actor, z_critic

    def __len__(self):
        return self.x.shape[0]


# ----------------------------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------------------------
class ContinualStreamAC(nn.Module):
    def __init__(self, task_dims, hidden_size=128, lr_policy=1e-4, lr_value=2e-4, gamma=0.99, lamda=0.8,
                 beta=0.99995, sparsity=0.95, rep_lambda=1.0, rep_batch_size=32):
        super().__init__()
        self.gamma, self.lamda, self.beta = gamma, lamda, beta
        self.lr_policy, self.lr_value = lr_policy, lr_value
        self.rep_lambda, self.rep_batch_size = rep_lambda, rep_batch_size
        self.actor_in = nn.ModuleList([InputLayer(o, hidden_size) for o, _ in task_dims])
        self.critic_in = nn.ModuleList([InputLayer(o, hidden_size) for o, _ in task_dims])
        self.actor_trunk = Trunk(hidden_size)
        self.critic_trunk = Trunk(hidden_size)
        self.actor_head = nn.ModuleList([GaussianHead(hidden_size, a) for _, a in task_dims])
        self.critic_head = nn.ModuleList([ValueHead(hidden_size) for _ in task_dims])
        sparse_init(self.modules(), target_sparsity=sparsity)
        self.memories = []
        self.task = None

    def set_task(self, k):
        """Train input_k + shared trunk + head_k. Modules of other tasks are frozen."""
        self.task = k
        for i in range(len(self.actor_in)):
            for m in (self.actor_in[i], self.critic_in[i], self.actor_head[i], self.critic_head[i]):
                m.requires_grad_(i == k)
        self.policy_params = list(self.actor_in[k].parameters()) + list(self.actor_trunk.parameters()) + list(self.actor_head[k].parameters())
        self.value_params = list(self.critic_in[k].parameters()) + list(self.critic_trunk.parameters()) + list(self.critic_head[k].parameters())
        self.optimizer_policy = RepPreservingStreamingOptimizer(self.policy_params, lr=self.lr_policy, gamma=self.gamma, lamda=self.lamda, beta=self.beta)
        self.optimizer_value = RepPreservingStreamingOptimizer(self.value_params, lr=self.lr_value, gamma=self.gamma, lamda=self.lamda, beta=self.beta)

    def actor_features(self, x, k):
        return self.actor_trunk(self.actor_in[k](x))

    def critic_features(self, x, k):
        return self.critic_trunk(self.critic_in[k](x))

    def pi(self, x, k=None):
        k = self.task if k is None else k
        return self.actor_head[k](self.actor_features(x, k))

    def v(self, x, k=None):
        k = self.task if k is None else k
        return self.critic_head[k](self.critic_features(x, k))

    @torch.no_grad()
    def sample_action(self, s, k=None):
        mu, std = self.pi(torch.as_tensor(s, dtype=torch.float32), k)
        return Normal(mu, std).sample().numpy()

    @torch.no_grad()
    def store_memory(self, k, x):
        """Store z_A = f_{theta_A}(x_A) for the given (already normalized) task-k inputs."""
        x = torch.as_tensor(x, dtype=torch.float32)
        self.memories.append(RepMemory(k, x, self.actor_features(x, k).clone(), self.critic_features(x, k).clone()))

    def rep_loss(self):
        """Monte-Carlo estimate of E_{x_A} ||f_theta(x_A) - z_A||^2 for actor and critic."""
        loss_actor, loss_critic = 0.0, 0.0
        for mem in self.memories:
            idx = torch.randint(len(mem), (self.rep_batch_size,))
            x = mem.x[idx]
            loss_actor = loss_actor + (self.actor_features(x, mem.task) - mem.z_actor[idx]).pow(2).sum(1).mean()
            loss_critic = loss_critic + (self.critic_features(x, mem.task) - mem.z_critic[idx]).pow(2).sum(1).mean()
        return loss_actor, loss_critic

    def update_params(self, s, a, r, s_prime, terminated, truncated, entropy_coeff):
        reset = terminated or truncated
        done_mask = 0.0 if terminated else 1.0
        s = torch.as_tensor(s, dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.float32)
        r = torch.as_tensor(r, dtype=torch.float32)
        s_prime = torch.as_tensor(s_prime, dtype=torch.float32)
        done_mask = torch.as_tensor(done_mask, dtype=torch.float32)

        v_s = self.v(s)
        with torch.no_grad():
            v_prime = self.v(s_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta = td_target - v_s
            delta_value = float(delta.item())
            delta_sign = 1.0 if delta_value > 0.0 else -1.0 if delta_value < 0.0 else 0.0

        mu, std = self.pi(s)
        dist = Normal(mu, std)
        log_prob_pi = dist.log_prob(a).sum()
        entropy_pi = entropy_coeff * dist.entropy().sum() * delta_sign
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        v_s.backward()
        (log_prob_pi + entropy_pi).backward()

        aux_policy, aux_value, rep_values = None, None, (0.0, 0.0)
        if self.memories and self.rep_lambda > 0.0:
            loss_actor, loss_critic = self.rep_loss()
            g_pol = torch.autograd.grad(self.rep_lambda * loss_actor, self.policy_params, allow_unused=True)
            g_val = torch.autograd.grad(self.rep_lambda * loss_critic, self.value_params, allow_unused=True)
            aux_policy = {p: g for p, g in zip(self.policy_params, g_pol) if g is not None}
            aux_value = {p: g for p, g in zip(self.value_params, g_val) if g is not None}
            rep_values = (float(loss_actor), float(loss_critic))

        self.optimizer_policy.step(delta_value, reset=reset, aux_grads=aux_policy)
        self.optimizer_value.step(delta_value, reset=reset, aux_grads=aux_value)
        return rep_values


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def make_env(env_name, render=False):
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env


def frozen_normalize(normalizer, obs):
    """Normalize with the current statistics WITHOUT updating them."""
    stats = normalizer.obs_stats
    return (np.asarray(obs, dtype=np.float32) - stats.mean) / np.sqrt(stats.var + normalizer.epsilon)


def evaluate(agent, k, env_name, normalizer, episodes, seed):
    """Average return of task k's policy (input_k + current trunk + head_k), frozen task-k normalizer."""
    env = make_env(env_name)
    rets = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            s, _, terminated, truncated, info = env.step(agent.sample_action(frozen_normalize(normalizer, s), k))
            done = terminated or truncated
        rets.append(float(info['episode']['r'][0]))
    env.close()
    return rets


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(env_names, seed, lr_policy, lr_value, gamma, lamda, beta, steps_per_task, entropy_coeff, hidden_size,
         rep_lambda, memory_size, memory_window, rep_batch_size, eval_every, eval_episodes,
         debug, render=False, sparsity=0.95):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    envs = [make_env(n, render) for n in env_names]
    task_dims = [(e.observation_space.shape[0], e.action_space.shape[0]) for e in envs]
    agent = ContinualStreamAC(task_dims, hidden_size=hidden_size, lr_policy=lr_policy, lr_value=lr_value, gamma=gamma,
                              lamda=lamda, beta=beta, sparsity=sparsity, rep_lambda=rep_lambda, rep_batch_size=rep_batch_size)
    tag = "_to_".join(env_names)
    save_dir = "data_stream_ac_rpcl_{}_steps{}_rep_lambda{}_mem{}_window{}_rep_batch{}_lr_policy{}_lr_value{}_gamma{}_lamda{}_beta{}_entropy_coeff{}_hidden_size{}_sparsity{}".format(
        tag, steps_per_task, rep_lambda, memory_size, memory_window, rep_batch_size, lr_policy, lr_value, gamma, lamda, beta, entropy_coeff, hidden_size, sparsity)
    os.makedirs(save_dir, exist_ok=True)
    normalizers = []
    evals = {}  # (old_task, new_task) -> list of (step_in_new_task, returns)

    for k, (env_name, env) in enumerate(zip(env_names, envs)):
        agent.set_task(k)
        reward_scaler = ScaleReward(gamma=gamma)
        normalizer = NormalizeObservation(env.observation_space)
        normalizers.append(normalizer)
        if debug:
            print("seed: {} | task {}: {} | obs {} act {} | memories: {}".format(seed, k, env_name, *task_dims[k], len(agent.memories)), flush=True)

        def run_evals(step):
            for j in range(k):
                rets = evaluate(agent, j, env_names[j], normalizers[j], eval_episodes, seed=10_000 + seed)
                evals.setdefault((j, k), []).append((step, rets))
                print("[eval] task {} step {} | old task {} ({}) return {:.1f}".format(k, step, j, env_names[j], np.mean(rets)), flush=True)

        run_evals(0)
        returns, term_time_steps = [], []
        mem_raw, mem_seen = [], 0  # reservoir sample of raw observations from the final window
        window_start = steps_per_task - memory_window + 1
        rep_a = rep_c = 0.0
        s_raw, _ = env.reset(seed=seed + k)
        s = normalizer.normalize(s_raw)
        for step in range(1, steps_per_task + 1):
            if step >= window_start and memory_size > 0:
                mem_seen += 1
                if len(mem_raw) < memory_size:
                    mem_raw.append(np.asarray(s_raw, dtype=np.float32))
                else:
                    j = rng.integers(mem_seen)
                    if j < memory_size:
                        mem_raw[j] = np.asarray(s_raw, dtype=np.float32)
            a = agent.sample_action(s)
            s_prime_raw, r, terminated, truncated, info = env.step(a)
            s_prime = normalizer.normalize(s_prime_raw)
            r = reward_scaler.scale(r, terminated=terminated, truncated=truncated)
            rep_a, rep_c = agent.update_params(s, a, r, s_prime, terminated, truncated, entropy_coeff)
            s, s_raw = s_prime, s_prime_raw
            if terminated or truncated:
                if debug:
                    print("Task {} ({}) Episodic Return: {}, Time Step {}, L_rep actor {:.4f} critic {:.4f}".format(
                        k, env_name, info['episode']['r'][0], step, rep_a, rep_c), flush=True)
                returns.append(info['episode']['r'][0])
                term_time_steps.append(step)
                s_raw, _ = env.reset()
                s = normalizer.normalize(s_raw)
            if eval_every > 0 and k > 0 and step % eval_every == 0:
                run_evals(step)
        env.close()

        # Per-task results in plot.py's format (steps relative to the start of the task).
        task_dir = os.path.join(save_dir, "task{}_{}".format(k, env_name))
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "seed_{}.pkl".format(seed)), "wb") as f:
            pickle.dump((returns, term_time_steps, env_name), f)
        torch.save(agent.state_dict(), os.path.join(save_dir, "model_seed_{}_after_task{}.pt".format(seed, k)))

        # Store (x_A, z_A) with the final task-k normalizer and parameters theta_A.
        if k < len(env_names) - 1 and mem_raw:
            x = np.stack([frozen_normalize(normalizer, o) for o in mem_raw])
            agent.store_memory(k, x)
            if debug:
                print("Stored {} representations for task {} ({})".format(len(x), k, env_name), flush=True)

    with open(os.path.join(save_dir, "old_task_evals_seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump({"env_names": env_names, "evals": evals}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ) with representation-preserving continual learning')
    parser.add_argument('--env_names', type=str, nargs='+', default=['Humanoid-v4', 'Hopper-v4'], help='Tasks in training order')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr_policy', type=float, default=1e-4)
    parser.add_argument('--lr_value', type=float, default=2e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=0.99995, help='Retained fraction in [0, 1)')
    parser.add_argument('--steps_per_task', type=int, default=20_000_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--rep_lambda', type=float, default=1.0, help='Representation-preservation strength λ (0 = no penalty)')
    parser.add_argument('--memory_size', type=int, default=5000, help='Number of stored old-task inputs x_A')
    parser.add_argument('--memory_window', type=int, default=1_000_000, help='x_A are reservoir-sampled from the last N steps of each task')
    parser.add_argument('--rep_batch_size', type=int, default=32, help='Memory samples per step for estimating L_rep')
    parser.add_argument('--eval_every', type=int, default=500_000, help='Evaluate old tasks every N steps of a new task (0 = off)')
    parser.add_argument('--eval_episodes', type=int, default=5)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--sparsity', type=float, default=0.95, help='Global target sparsity for ERK initialization (fraction of weights set to zero)')
    args = parser.parse_args()
    main(args.env_names, args.seed, args.lr_policy, args.lr_value, args.gamma, args.lamda, args.beta, args.steps_per_task,
         args.entropy_coeff, args.hidden_size, args.rep_lambda, args.memory_size, args.memory_window, args.rep_batch_size,
         args.eval_every, args.eval_episodes, args.debug, args.render, args.sparsity)
