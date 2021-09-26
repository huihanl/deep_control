import argparse
import copy
import math
import os
from itertools import chain

import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import tqdm

from . import envs, nets, replay, run, utils

import sys

path = sys.argv[1]

state_space = 19
action_space = 7
log_std_low = -10
log_std_high = 2

# create agent
agent = dc.awac.AWACAgent(
    state_space,
    action_space,
    log_std_low,
    log_std_high,
)

actor_path = os.path.join(path, "actor.pt")
agent.load_state_dict(torch.load(actor_path))

mean_return = run.evaluate_agent(
    agent, test_env, eval_episodes, max_episode_steps, render
)