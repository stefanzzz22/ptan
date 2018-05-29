import gym
import ptan
import argparse
import numpy as np
import sys

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from lib import dqn_model, config, common, demo_data_reader
from collections import deque

PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000
# prioritized replay constants
e_prio = 0.001

params_save_file = "saved_models/model-prio-replay"

last_dq_losses = deque(maxlen=100)

def calc_loss(batch, batch_weights, net, tgt_net, gamma, cuda=False):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)

    states_v = Variable(torch.from_numpy(states))
    next_states_v = Variable(torch.from_numpy(next_states), volatile=True)
    actions_v = Variable(torch.from_numpy(actions))
    rewards_v = Variable(torch.from_numpy(rewards))
    done_mask = torch.ByteTensor(dones)
    batch_weights_v = Variable(torch.from_numpy(batch_weights))
    if cuda:
        states_v = states_v.cuda(async=True)
        next_states_v = next_states_v.cuda(async=True)
        actions_v = actions_v.cuda(async=True)
        rewards_v = rewards_v.cuda(async=True)
        done_mask = done_mask.cuda(async=True)
        batch_weights_v = batch_weights_v.cuda(async=True)

    state_all_action_values = net(states_v)
    state_action_values = state_all_action_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_actions = net(next_states_v).max(1)[1]
    next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    next_state_values[done_mask] = 0.0
    next_state_values.volatile = False

    # DQN Loss
    expected_state_action_values = next_state_values * gamma + rewards_v
    dq_losses = nn.SmoothL1Loss(reduce=False)(state_action_values, expected_state_action_values)
    dq_loss = (batch_weights_v * dq_losses).sum()

    last_dq_losses.append(dq_loss.data.cpu().numpy() / len(batch))

    return dq_loss, (dq_losses.data.abs() + e_prio).cpu().numpy()


def main():
    global params_save_file

    game = 'spaceinvaders'
    params_save_file += '-' + game

    params = config.HYPERPARAMS[game]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env, skip=params['skip-frames'])

    print("Parameters:")
    print(params)
    sys.stdout.flush()

    writer = SummaryWriter(comment="-" + params['run_name'] + "-prio-replay")
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    if args.cuda:
        net.cuda()

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, cuda=args.cuda)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'],
                    steps_count=1)
    buffer = ptan.experience.PrioritizedReplayBuffer(exp_source, params['replay_size'], PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    beta = BETA_START

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += params['steps']
            buffer.populate(params['steps'])
            epsilon_tracker.frame(frame_idx)
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                writer.add_scalar("beta", beta, frame_idx)
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon, last_dq_losses):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(params['batch_size'] * params['steps'], beta)
            loss_v, sample_prios = calc_loss(batch, batch_weights, net, tgt_net.target_model,
                                                params["gamma"], cuda=args.cuda)
            loss_v.backward()
            optimizer.step()
            buffer.update_priorities(batch_indices, sample_prios)

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()

            if frame_idx % params['save_params_every'] == 0:
                torch.save(net.state_dict(), params_save_file + str(frame_idx))

    torch.save(net.state_dict(), params_save_file + str(frame_idx))

if __name__ == "__main__":
    main()