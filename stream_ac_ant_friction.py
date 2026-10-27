"""
Stream AC(λ) on the alternating-friction Ant-v4 task (Section 4.1 of
"Streaming Deep Reinforcement Learning Finally Works", arXiv:2410.14606).

The coefficient of friction between the robot and the floor alternates between
a high-traction value (default 2.0) and a slippery value (default 0.02) every
`switch_every` steps (default 2M), giving ten regimes over 20M steps. Each
switch is instantaneous (it can happen mid-episode) and the agent receives no
signal about it.
"""
import argparse
import os
import pickle
import torch
import numpy as np
import gymnasium as gym
from obs_reward_transforms import NormalizeObservation, ScaleReward
from stream_ac_continuous import StreamAC


def set_friction(env, mu):
    """
    Set the sliding friction coefficient of every geom in the MuJoCo model.
    MuJoCo combines the friction of two contacting geoms by taking the maximum,
    so changing only the floor would not make it slippery; all geoms are set.
    Torsional/rolling coefficients are left unchanged.
    """
    env.unwrapped.model.geom_friction[:, 0] = mu


def friction_at(step, switch_every, high, low):
    """Regime 0 (steps 1..switch_every) is high friction, then alternates."""
    return high if ((step - 1) // switch_every) % 2 == 0 else low


def main(env_name, seed, lr_policy, lr_value, gamma, lamda, beta, total_steps, entropy_coeff, hidden_size,
         switch_every, high_friction, low_friction, debug, render=False, sparsity=0.95):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    reward_scaler = ScaleReward(gamma=gamma)
    observation_normalizer = NormalizeObservation(env.observation_space)
    agent = StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], hidden_size=hidden_size, lr_policy=lr_policy, lr_value=lr_value, gamma=gamma, lamda=lamda, beta=beta, sparsity=sparsity)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env.spec.id))
    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    current_mu = friction_at(1, switch_every, high_friction, low_friction)
    set_friction(env, current_mu)
    if debug:
        print("Friction set to {} at time step 1".format(current_mu))
    s = observation_normalizer.normalize(s)
    for step in range(1, total_steps + 1):
        mu = friction_at(step, switch_every, high_friction, low_friction)
        if mu != current_mu:
            current_mu = mu
            set_friction(env, current_mu)
            if debug:
                print("Friction switched to {} at time step {}".format(current_mu, step))
        a = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        s_prime = observation_normalizer.normalize(s_prime)
        r = reward_scaler.scale(r, terminated=terminated, truncated=truncated)
        agent.update_params(s, a, r, s_prime, terminated, truncated, entropy_coeff)
        s = s_prime
        if terminated or truncated:
            if debug:
                print("Episodic Return: {}, Time Step {}, Friction {}".format(info['episode']['r'][0], step, current_mu))
            returns.append(info['episode']['r'][0])
            term_time_steps.append(step)
            s, _ = env.reset()
            s = observation_normalizer.normalize(s)
    env.close()
    save_dir = "data_stream_ac_altfriction_{}_high{}_low{}_switch{}_lr_policy{}_lr_value{}_gamma{}_lamda{}_beta{}_entropy_coeff{}_hidden_size{}_sparsity{}".format(env.spec.id, high_friction, low_friction, switch_every, lr_policy, lr_value, gamma, lamda, beta, entropy_coeff, hidden_size, sparsity)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Same (returns, term_time_steps, env_name) format as the other scripts so plot.py works unchanged.
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, "{} (alternating friction)".format(env_name)), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ) on alternating-friction Ant')
    parser.add_argument('--env_name', type=str, default='Ant-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr_policy', type=float, default=1e-4)
    parser.add_argument('--lr_value', type=float, default=2e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=0.99995, help='Retained fraction in [0, 1)')
    parser.add_argument('--total_steps', type=int, default=20_000_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--switch_every', type=int, default=2_000_000, help='Number of steps between friction switches')
    parser.add_argument('--high_friction', type=float, default=2.0, help='High-traction friction coefficient')
    parser.add_argument('--low_friction', type=float, default=0.02, help='Slippery friction coefficient')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--sparsity', type=float, default=0.95, help='Global target sparsity for ERK initialization (fraction of weights set to zero)')
    args = parser.parse_args()
    main(args.env_name, args.seed, args.lr_policy, args.lr_value, args.gamma, args.lamda, args.beta, args.total_steps, args.entropy_coeff, args.hidden_size,
         args.switch_every, args.high_friction, args.low_friction, args.debug, args.render, args.sparsity)
