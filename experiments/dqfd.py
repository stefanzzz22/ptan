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
LOSS_LAMBDAS = [1., 1., 1.]
L2_REG_LAMBDA = 1e-5
MARGIN_LOSS = 0.8
# prioritized replay constants
e_non_demo = 0.001
e_demo = 1

params_save_file = "saved_models/model-dqfd"

last_dq_losses = deque(maxlen=100)
last_n_losses = deque(maxlen=100)
last_e_losses = deque(maxlen=100)
last_demo_sizes = deque(maxlen=100)

def calc_loss(batch, demo_mask, batch_weights, net, tgt_net, gamma, gamma_n, cuda=False):
    states, actions, rewards, dones, next_states, rewards_n, dones_n, last_states = common.unpack_batch_dqfd(batch)

    demo_mask = torch.ByteTensor(demo_mask)
    batch_weights_v = Variable(torch.from_numpy(batch_weights))

    states_v = Variable(torch.from_numpy(states))
    next_states_v = Variable(torch.from_numpy(next_states), volatile=True)
    last_states_v = Variable(torch.from_numpy(last_states), volatile=True)
    actions_v = Variable(torch.from_numpy(actions))
    rewards_v = Variable(torch.from_numpy(rewards))
    rewards_n_v = Variable(torch.from_numpy(rewards_n))
    done_mask = torch.ByteTensor(dones)
    done_n_mask = torch.ByteTensor(dones_n)

    if cuda:
        demo_mask = demo_mask.cuda(async=True)
        batch_weights_v = batch_weights_v.cuda(async=True)

        states_v = states_v.cuda(async=True)
        next_states_v = next_states_v.cuda(async=True)
        last_states_v = last_states_v.cuda(async=True)
        actions_v = actions_v.cuda(async=True)
        rewards_v = rewards_v.cuda(async=True)
        rewards_n_v = rewards_n_v.cuda(async=True)
        done_mask = done_mask.cuda(async=True)
        done_n_mask = done_n_mask.cuda(async=True)

    state_all_action_values = net(states_v)
    state_action_values = state_all_action_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    next_state_actions = net(next_states_v).max(1)[1]
    next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    next_state_values[done_mask] = 0.0
    next_state_values.volatile = False

    last_state_actions = net(last_states_v).max(1)[1]
    last_state_values = tgt_net(last_states_v).gather(1, last_state_actions.unsqueeze(-1)).squeeze(-1)
    last_state_values[done_n_mask] = 0.0
    last_state_values.volatile = False

    # TD(1) Loss
    expected_state_action_values = next_state_values * gamma + rewards_v
    dq_losses = nn.SmoothL1Loss(reduce=False)(state_action_values, expected_state_action_values)
    dq_loss = (batch_weights_v * dq_losses).sum()

    # TD(n) loss
    expected_n_state_action_values = last_state_values * gamma_n + rewards_n_v
    n_losses = nn.SmoothL1Loss(reduce=False)(state_action_values, expected_n_state_action_values)
    n_loss = (batch_weights_v * n_losses).sum()

    # Supervised loss
    e_loss = 0
    demo_size = demo_mask.sum()
    if demo_size > 0:
        demo_action_values = state_all_action_values[np.arange(state_all_action_values.shape[0]), actions_v]
        action_values = state_all_action_values + MARGIN_LOSS
        action_values[np.arange(action_values.shape[0]), actions_v] = demo_action_values
        e_losses = action_values.max(1)[0] - demo_action_values
        e_losses *= batch_weights_v
        e_losses[~demo_mask] = 0
        e_loss = e_losses.sum()

    tot_loss = sum([l * loss for l, loss in zip(LOSS_LAMBDAS, [dq_loss, n_loss, e_loss])])

    sample_prios = torch.zeros(len(demo_mask))
    if cuda:
        sample_prios = sample_prios.cuda(async=True)
    sample_prios[demo_mask] += e_demo
    sample_prios[~demo_mask] += e_non_demo

    last_dq_losses.append(dq_loss.data.cpu().numpy() / len(batch))
    last_n_losses.append(n_loss.data.cpu().numpy() / len(batch))
    last_e_losses.append(e_loss.data.cpu().numpy() / demo_size if demo_size > 0 else 0)
    last_demo_sizes.append(demo_size)

    return tot_loss, (dq_losses.data.abs() + sample_prios).cpu().numpy()


def main():
    global params_save_file

    game = 'revenge'
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

    writer = SummaryWriter(comment="-" + params['run_name'] + "-dqfd(PDD DQN)")
    net = dqn_model.DuelingDQN(env.observation_space.shape, env.action_space.n)
    if args.cuda:
        net.cuda()

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, cuda=args.cuda)

    demo_data = demo_data_reader.get_demo_data(env, game, num_states=params['demo_size'], skip=params['skip-frames'])
    exp_source = ptan.experience.ExperienceSourceNFirstLast(env, agent, gamma=params['gamma'],
                    steps_count=params['n-steps'], demo_data=demo_data)
    buffer = ptan.experience.PrioritizedReplayBuffer(exp_source, params['replay_size'], PRIO_REPLAY_ALPHA)
    buffer.populate_demo_data()
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'], weight_decay=L2_REG_LAMBDA)

    print("Demo data size: {}".format(buffer.demo_samples))
    sys.stdout.flush()

    frame_idx = 0
    beta = BETA_START

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += params['steps']
            if frame_idx > params['pretrain_steps']:
                buffer.populate(params['steps'])
            else:
                if frame_idx % 500 == 0:
                    writer.add_scalar("beta", beta, frame_idx)
                    reward_tracker.record_training(frame_idx, selector.epsilon, last_dq_losses, last_n_losses,
                        last_e_losses, last_demo_sizes)

            epsilon_tracker.frame(frame_idx)
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                writer.add_scalar("beta", beta, frame_idx)
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon, last_dq_losses,
                    last_n_losses, last_e_losses, last_demo_sizes):
                    break

            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(params['batch_size'] * params['steps'], beta)
            batch_demo_mask = (np.array(batch_indices) < buffer.demo_samples).astype(np.uint8)

            loss_v, sample_prios = calc_loss(batch, batch_demo_mask, batch_weights, net, tgt_net.target_model,
                                                params["gamma"], params["gamma"] ** params['n-steps'],
                                                cuda=args.cuda)
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