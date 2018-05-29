import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def unpack_batch_dqfd(batch):
    states, actions, rewards, dones, next_states, rewards_n, dones_n, last_states = [], [], [], [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.next_state is None)
        if exp.next_state is None:
            next_states.append(state)       # the result will be masked anyway
        else:
            next_states.append(np.array(exp.next_state, copy=False))
        rewards_n.append(exp.reward_n)
        dones_n.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))

    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(next_states, copy=False), \
           np.array(rewards_n, dtype=np.float32), np.array(dones_n, dtype=np.uint8), \
           np.array(last_states, copy=False)


def unpack_batch(batch):
    states, actions, rewards, dones, next_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            next_states.append(state)       # the result will be masked anyway
        else:
            next_states.append(np.array(exp.last_state, copy=False))

    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(next_states, copy=False)



class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None, last_dq_losses=None, last_n_losses=None,
                last_e_losses=None, last_demo_sizes=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward_100 %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if last_dq_losses is not None and len(last_dq_losses) > 0:
            self.writer.add_scalar("dq_loss_100", np.mean(last_dq_losses), frame)
        if last_n_losses is not None:
            self.writer.add_scalar("n_loss_100", np.mean(last_n_losses), frame)
        if last_e_losses is not None:
            self.writer.add_scalar("e_loss_100", np.mean(last_e_losses), frame)
        if last_demo_sizes is not None:
            self.writer.add_scalar("demo_size_100", np.mean(last_demo_sizes), frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False

    def record_training(self, frame, epsilon=None, last_dq_losses=None, last_n_losses=None,
                        last_e_losses=None, last_demo_sizes=None):
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        print("%d (training): speed %.2f f/s" % (frame, speed))
        sys.stdout.flush()

        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        if last_dq_losses is not None:
            self.writer.add_scalar("dq_loss_100", np.mean(last_dq_losses), frame)
        if last_n_losses is not None:
            self.writer.add_scalar("n_loss_100", np.mean(last_n_losses), frame)
        if last_e_losses is not None:
            self.writer.add_scalar("e_loss_100", np.mean(last_e_losses), frame)
        if last_demo_sizes is not None:
            self.writer.add_scalar("demo_size_100", np.mean(last_demo_sizes), frame)


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)
