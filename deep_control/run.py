import gym
import numpy as np
import torch
import os 

from . import envs, utils

# TODO: build wrapper env instead of using this function
def process_robomimic_state(state):
    ee_pos = state["robot0_eef_pos"]
    ee_quat = state["robot0_eef_quat"]
    gripper_pos = state["robot0_gripper_qpos"]
    object_info = state["object"]
    obs = np.concatenate([ee_pos, ee_quat, gripper_pos, object_info])
    return obs

def save_video(video_path, video_name, img_lst):
    import os
    import imageio
    video_writer = imageio.get_writer(os.path.join(video_path, video_name), fps=20)
    for img in img_lst:
        video_writer.append_data(img)
    video_writer.close()

def run_env(agent, env, episodes, max_steps, curr_step, render=False, verbosity=1, discount=1.0):
    episode_return_history = []
    done_history = []
    for episode in range(episodes):
        print("At episode {} / {}".format(episode, episodes))
        episode_return = 0.0
        state = env.reset()

        print("original state: ", state)
        
        state = process_robomimic_state(state)

        img_lst = [] # save videos
        done, info = False, {}
        for step_num in range(100):
            if done:
                break
            print("state received: ", state)
            action = agent.forward(state)
            print("action: ", action)
            state, reward, done, info = env.step(action)
            state = process_robomimic_state(state)
            img_lst.append(env.render(mode="rgb_array", height=256, width=256, camera_name="frontview")) # TODO
            episode_return += reward
        if verbosity:
            print(f"Episode {episode}:: {episode_return}")
        episode_return_history.append(episode_return)
        done_history.append(float(int(done)))
        print("success: ", done, " returns: ", episode_return)
        
        video_path = "/home/huihanl/test_video/step_{}".format(curr_step)
        print("os.path.exists(video_path): ", os.path.exists(video_path))
        print("video path: ", video_path)
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        save_video(video_path, "ep_{}.mp4".format(episode), img_lst)

    return torch.tensor(episode_return_history), torch.tensor(done_history)


def exploration_noise(action, random_process):
    return np.clip(action + random_process.sample(), -1.0, 1.0)


def evaluate_agent(
    agent, env, eval_episodes, max_episode_steps, curr_step, render=False, verbosity=0
):
    agent.eval()
    returns, successes = run_env(
        agent, env, eval_episodes, max_episode_steps, curr_step, render, verbosity=verbosity
    )
    agent.train()
    mean_return = returns.mean()
    success_return = successes.mean()
    return mean_return, success_return


def collect_experience_by_steps(
    agent,
    env,
    buffer,
    num_steps,
    current_state=None,
    current_done=None,
    steps_this_ep=None,
    max_rollout_length=None,
    random_process=None,
):
    if current_state is None:
        state = env.reset()
    else:
        state = current_state
    if current_done is None:
        done = False
    else:
        done = current_done
    if steps_this_ep is None:
        steps_this_ep = 0
    for step in range(num_steps):
        if done:
            state = env.reset()
            steps_this_ep = 0

        # collect a new transition
        action = agent.collection_forward(state)
        if random_process is not None:
            action = exploration_noise(action, random_process, env.action_space.high[0])
        next_state, reward, done, info = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state

        steps_this_ep += 1
        if max_rollout_length and steps_this_ep >= max_rollout_length:
            done = True
    return state, done, steps_this_ep


def collect_experience_by_rollouts(
    agent,
    env,
    buffer,
    num_rollouts,
    max_rollout_length,
    random_process=None,
):
    for rollout in range(num_rollouts):
        state = env.reset()
        done = False
        step_num = 0
        while not done:
            action = agent.collection_forward(state)
            if random_process is not None:
                action = exploration_noise(
                    action, random_process, env.action_space.high[0]
                )
            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            step_num += 1
            if step_num >= max_rollout_length:
                done = True


def warmup_buffer(buffer, env, warmup_steps, max_episode_steps):
    # use warmp up steps to add random transitions to the buffer
    state = env.reset()
    done = False
    steps_this_ep = 0
    for _ in range(warmup_steps):
        if done:
            state = env.reset()
            steps_this_ep = 0
            done = False
        rand_action = env.action_space.sample()
        if not isinstance(rand_action, np.ndarray):
            rand_action = np.array(float(rand_action))
        next_state, reward, done, info = env.step(rand_action)
        buffer.push(state, rand_action, reward, next_state, done)
        state = next_state
        steps_this_ep += 1
        if steps_this_ep >= max_episode_steps:
            done = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--env", type=str)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--save", type=str)
    parser.add_argument("--algo", type=str)
    parser.add_argument("--max_steps", type=int, default=300)
    args = parser.parse_args()

    agent, env = envs.load_env(args.env, args.algo)
    agent.load(args.agent)
    run_env(agent, env, args.episodes, args.max_steps, args.render, verbosity=1)
