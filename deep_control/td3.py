import argparse
import time
import copy

import torch
import torch.nn.functional as F
import numpy as np
import gym

from . import utils
from . import run


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def td3(agent, train_env, args):
    agent.to(device)

    # initialize target networks
    target_agent = copy.deepcopy(agent)
    target_agent.to(device)
    utils.hard_update(target_agent.actor, agent.actor)
    utils.hard_update(target_agent.critic1, agent.critic1)
    utils.hard_update(target_agent.critic2, agent.critic2)

    random_process = utils.OrnsteinUhlenbeckProcess(size=train_env.action_space.shape, sigma=args.sigma_start, sigma_min=args.sigma_final, n_steps_annealing=args.sigma_anneal, theta=args.theta)

    buffer = utils.ReplayBuffer(args.buffer_size)
    critic1_optimizer = torch.optim.Adam(agent.critic1.parameters(), lr=args.critic_lr, weight_decay=args.critic_l2)
    critic2_optimizer = torch.optim.Adam(agent.critic2.parameters(), lr=args.critic_lr, weight_decay=args.critic_l2)
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=args.actor_lr, weight_decay=args.actor_l2)

    save_dir = utils.make_process_dirs(args.name)
    test_env = copy.deepcopy(train_env)

    state = train_env.reset()
    done = False
    for _ in range(args.warmup_steps):
        if done: state = train_env.reset(); done = False
        rand_action = train_env.action_space.sample()
        next_state, reward, done, info = train_env.step(rand_action)
        buffer.push(state, rand_action, reward, next_state, done)
        state = next_state

    step_count = 0
    for episode in range(args.num_episodes):
        rollout = utils.collect_rollout(agent, random_process, train_env, args)

        for (state, action, rew, next_state, done, info) in rollout:
            buffer.push(state, action, rew, next_state, done)

        for optimization_step in range(args.opt_steps):
            update_policy = (step_count % args.delay == 0)
            _td3_learn(args, buffer, target_agent, agent, actor_optimizer, critic1_optimizer, critic2_optimizer, update_policy)

            # move target model towards training model
            if update_policy:
                utils.soft_update(target_agent.actor, agent.actor, args.tau)
            utils.soft_update(target_agent.critic1, agent.critic1, args.tau)
            utils.soft_update(target_agent.critic2, agent.critic2, args.tau)
            step_count += 1
        
        if episode % args.eval_interval == 0:
            mean_return = utils.evaluate_agent(agent, test_env, args)
            print(f"Episodes of training: {episode+1}, mean reward in test mode: {mean_return}")
   
    agent.save(save_dir)
    return agent

def _td3_learn(args, buffer, target_agent, agent, actor_optimizer, critic1_optimizer, critic2_optimizer, update_policy=True):
    batch = buffer.sample(args.batch_size)
    # batch will be None if not enough experience has been collected yet
    if not batch:
        return
    
    # prepare transitions for models
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
    
    cat_tuple = lambda t : torch.cat(t).to(device)
    list_to_tensor = lambda t : torch.tensor(t).unsqueeze(0).to(device)
    state_batch = cat_tuple(state_batch)
    next_state_batch = cat_tuple(next_state_batch)
    action_batch = cat_tuple(action_batch)
    reward_batch = list_to_tensor(reward_batch).T
    done_batch = list_to_tensor(done_batch).T

    agent.train()

    with torch.no_grad():
        # create critic targets (clipped double Q learning)
        target_action_s2 = target_agent.actor(next_state_batch)
        # target smoothing
        target_action_s2 += torch.clamp(args.target_noise_scale*torch.randn(*target_action_s2.shape), -args.c, args.c)
        target_action_value_s2 = torch.min(target_agent.critic1(next_state_batch, target_action_s2), target_agent.critic2(next_state_batch, target_action_s2))
        td_target = reward_batch + args.gamma*(1.-done_batch)*target_action_value_s2

    # update first critic
    agent_critic1_pred = agent.critic1(state_batch, action_batch)
    critic1_loss = F.mse_loss(td_target, agent_critic1_pred)
    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    if args.critic_clip:
        torch.nn.utils.clip_grad_norm_(agent.critic1.parameters(), args.critic_clip)
    critic1_optimizer.step()

    # update second critic
    agent_critic2_pred = agent.critic2(state_batch, action_batch)
    critic2_loss = F.mse_loss(td_target, agent_critic2_pred)
    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    if args.critic_clip:
        torch.nn.utils.clip_grad_norm_(agent.critic2.parameters(), args.critic_clip)
    critic2_optimizer.step()

    if update_policy:
        # actor update
        agent_actions = agent.actor(state_batch)
        actor_loss = -agent.critic1(state_batch, agent_actions).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        if args.actor_clip:
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), args.actor_clip)
        actor_optimizer.step()

def parse_args():
    parser = argparse.ArgumentParser(description='Train agent with DDPG')
    parser.add_argument('--env', type=str, default='Pendulum-v0', help='training environment')
    parser.add_argument('--num_episodes', type=int, default=500,
                        help='number of episodes for training')
    parser.add_argument('--max_episode_steps', type=int, default=250,
                        help='maximum steps per episode')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='training batch size')
    parser.add_argument('--tau', type=float, default=.001,
                        help='for model parameter % update')
    parser.add_argument('--actor_lr', type=float, default=1e-4,
                        help='actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=1e-3,
                        help='critic learning rate')
    parser.add_argument('--gamma', type=float, default=.99,
                        help='gamma, the discount factor')
    parser.add_argument('--sigma_final', type=float, default=.2)
    parser.add_argument('--sigma_anneal', type=float, default=10000, help='How many steps to anneal sigma over.')
    parser.add_argument('--theta', type=float, default=.15,
        help='theta for Ornstein Uhlenbeck process computation')
    parser.add_argument('--sigma_start', type=float, default=.2,
        help='sigma for Ornstein Uhlenbeck process computation')
    parser.add_argument('--buffer_size', type=int, default=100000,
        help='replay buffer size')
    parser.add_argument('--eval_interval', type=int, default=15,
        help='how often to test the agent without exploration (in episodes)')
    parser.add_argument('--eval_episodes', type=int, default=10,
        help='how many episodes to run for when testing')
    parser.add_argument('--warmup_steps', type=int, default=1000,
        help='warmup length, in steps')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--actor_clip', type=float, default=None)
    parser.add_argument('--critic_clip', type=float, default=None)
    parser.add_argument('--name', type=str, default='ddpg_run')
    parser.add_argument('--opt_steps', type=int, default=50)
    parser.add_argument('--actor_l2', type=float, default=0.)
    parser.add_argument('--critic_l2', type=float, default=1e-4)
    parser.add_argument('--delay', type=int, default=2)
    parser.add_argument('--target_noise_scale', type=float, default=.2)
    parser.add_argument('--c', type=float, default=.5)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    agent, env = run.load_env(args.env, 'td3')
    agent = td3(agent, env, args)

