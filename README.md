
# PTAN

PTAN stands for PyTorch AgentNet -- reimplementation of
[AgentNet](https://github.com/yandexdataschool/AgentNet) library for
[PyTorch](http://pytorch.org/)

This library was used in ["Practical Deep Reinforcement Learning"](https://www.packtpub.com/big-data-and-business-intelligence/practical-deep-reinforcement-learning) book, here you can find [sample sources](https://github.com/PacktPublishing/Practical-Deep-Reinforcement-Learning).

## Installation

Install locally:
```bash
pip install -e .
```

## Requirements

* [PyTorch](http://pytorch.org/): version 0.3.1 is required
* [OpenAI Gym](https://gym.openai.com/): ```pip install gym gym[atari]```
* [Python OpenCV](https://pypi.org/project/opencv-python/): ```pip install opencv-python```
* [TensorBoard for PyTorch](https://github.com/lanpa/tensorboard-pytorch): ```pip install tensorboard-pytorch```

### Note for [Anaconda Python](https://anaconda.org/anaconda/python) users

To run some of the samples, you will need these modules:
```bash
conda install pytorch=0.3.1 torchvision -c pytorch
pip install tensorboard-pytorch
pip install gym
pip install gym[atari]
pip install opencv-python
```

### DQFD implementation details
A DQFD implementation can be found in `experiments/dqfd.py`, but it requires creating a new folder for demonstration
data `experiments/demo`. Demo data can be downloaded from https://github.com/yobibyte/atarigrandchallenge.

The `experiments/demo` directory should be structured similarly to the AtariGrandChallenge format:
```
demo/
├── revenge
│   ├── screens
│   │   └── revenge
│   │       ├── 1
│   │       │   ├── 1.png
│   │       │   └── 2.png
│   │       └── 2
│   │           ├── 1.png
│   │           └── 2.png
│   └── trajectories
│       └── revenge
│           ├── 1.txt
│           └── 2.txt
└── spaceinvaders
    ├── screens
    |     ....
    └── trajectories
          ....
```
