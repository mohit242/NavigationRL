try:
    import comet_ml
except ImportError:
    pass

import torch
import argparse
import numpy as np
import json
from collections import deque
from unityagents import UnityEnvironment
from navigationrl.agents import DQNAgent


if __name__ == "__main__":

    # Instantiate environment
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config", help="File path of config json file", type=str, default="config.json")
    argparser.add_argument("--play", help="sets mode to play instead of train", action="store_true")
    print("Loading params from config.json .....")
    args = argparser.parse_args()
    with open(args.config, 'r') as f:
        params = json.load(f)
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64", no_graphics=params['no_graphics'])
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # qnet_params = {'type': 'simple', 'fc1_units': 128, 'fc2_units': 64}
    qnet_params = params['qnet_params']

    logger = None
    if params['logging'] == True:
        logger = comet_ml.Experiment(project_name="Banana_Navigation", workspace="drl")
        logger.log_parameters(params)

    agent = DQNAgent(state_size, action_size, params['seed'], device, qnet_params, buffer_size=params['buffer_size'],
                     batch_size=params['batch_size'], gamma=params['gamma'], tau=params['tau'], lr=params['lr'],
                     update_every=params['update_every'], logger=logger)

    if not args.play:
        num_episodes = params['max_episodes']
        eps_start = params['eps_start']
        eps_end = params['eps_end']
        eps_decay = params['eps_decay']

        scores = []
        scores_window = deque(maxlen=100)
        eps = eps_start
        for i_episode in range(num_episodes):
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0
            while True:
                action = agent.act(state, eps)
                env_info = env.step(action)[brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            eps = max(eps_end, eps_decay*eps)
            if logger is not None:
                logger.log_metric("score", score)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= 13.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                             np.mean(scores_window)))
                break

        torch.save(agent.qnet_local.state_dict(), params["network_weight_file"])
        if logger is not None:
            logger.log_asset(params["network_weight_file"])
            logger.log_asset(args.config)

    else:
        print("Loading network weights from file - {}".format(params["network_weight_file"]))
        agent.qnet_local.load_state_dict(torch.load(params["network_weight_file"]))
        score = 0
        while True:
            action = agent.act(state, eps=0)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        print("Score {}".format(score))

