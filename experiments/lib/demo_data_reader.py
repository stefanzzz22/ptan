from lib.agc import dataset as ds
import gym
import os
import numpy as np
from collections import deque
from ptan.common.wrappers import LazyFrames
import sys

datadir = 'demo'

def get_action_id(env, action_name):
    try:
        idx = env.unwrapped.get_action_meanings().index(action_name)
    except ValueError:
        idx = 0

    return idx


def get_demo_data(env, game, num_states, skip=4, stack_frames=4):
    """
    Gets the demonstration data for a game
    :param game: string representing name of the game
    :returns array of (state, action, reward, done) where state has shape (stack_frames, 84, 84)
    """

    print("Importing training data")
    sys.stdout.flush()
    dataset = ds.AtariDataset(os.path.join(datadir, game))
    print("Trajectories imported")
    sys.stdout.flush()

    data = dataset.compile_data(game, max_nb_transitions=skip * num_states)

    print("Training data imported")
    print("Preparing training data...")
    sys.stdout.flush()

    obs_buffer = deque(maxlen=2)
    total_reward = 0.0
    max_action = 0 # NOOP
    demo_data = []
    for i, frame in enumerate(data):
        state, action, reward, done = frame['state'], frame['action'], frame['reward'], frame['terminal']
        obs_buffer.append(state)
        total_reward += reward
        max_action = max(max_action, get_action_id(env, action))

        if done or (i + 1) % skip == 0:
            max_state = np.max(np.stack(obs_buffer), axis=0)
            max_state = np.transpose(max_state, axes=(2, 0, 1))
            demo_data.append((max_state, max_action, total_reward, done))
            total_reward = 0.0
            max_action = 0

    data = []
    demo_data = demo_data[:num_states]

    # Stack frames
    frames = deque(maxlen=stack_frames)
    for _ in range(stack_frames):
        frames.append(demo_data[0][0])

    stacked_demo_data = []
    for (state, action, reward, done) in demo_data:
        frames.append(state)
        stacked_demo_data.append((LazyFrames(list(frames)), action, reward, done))

    print("Training data prepared")
    sys.stdout.flush()

    return stacked_demo_data


if __name__ == '__main__':
    env = gym.make('SpaceInvadersNoFrameskip-v4')
    get_demo_data(env, 'spaceinvaders', num_states=100)
