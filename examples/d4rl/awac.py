import argparse

import gym
import d4rl
import numpy as np

import deep_control as dc

import os
import json
import h5py
import argparse
import imageio
import numpy as np

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase
from copy import deepcopy
import robomimic.envs.env_base as EB

def get_env_metadata_from_dataset(dataset_path):
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    env_meta = json.loads(f["data"].attrs["env_args"])
    f.close()
    return env_meta

def create_env_from_metadata(
    env_meta,
    env_name=None,
    render=False,
    render_offscreen=False,
    use_image_obs=False,
):
    if env_name is None:
        env_name = env_meta["env_name"]
    env_type = get_env_type(env_meta=env_meta)
    env_kwargs = env_meta["env_kwargs"]

    env = create_env(
        env_type=env_type,
        env_name=env_name,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
        **env_kwargs,
    )
    return env

def create_env_from_dataset(dataset_path):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True)
    return env

##########################################
##########################################


def train_d4rl_awac(args):
    dset = np.load(args.dataset_path, allow_pickle=True).item()
       
    train_env = create_env_from_dataset(args.dataset_path_env) 
    test_env = create_env_from_dataset(args.dataset_path_env)
    
    state_space = args.state_space #test_env.observation_space
    action_space = args.action_space #test_env.action_space

    # create agent
    agent = dc.awac.AWACAgent(
        state_space,
        action_space,
        args.log_std_low,
        args.log_std_high,
    )

    # get offline datset
    #dset = d4rl.qlearning_dataset(test_env)
    dset_size = dset["observations"].shape[0]
    # create replay buffer
    buffer = dc.replay.PrioritizedReplayBuffer(
        size=dset_size,
        state_shape=(state_space,),
        state_dtype=float,
        action_shape=(action_space,),
    )
    buffer.load_experience(
        dset["observations"],
        dset["actions"],
        dset["rewards"],
        dset["next_observations"],
        dset["terminals"],
    )

    # run awac
    dc.awac.awac(
        agent=agent, train_env=train_env, test_env=test_env, buffer=buffer, **vars(args)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dc.envs.add_gym_args(parser)
    dc.awac.add_args(parser)
    args = parser.parse_args()
    train_d4rl_awac(args)
