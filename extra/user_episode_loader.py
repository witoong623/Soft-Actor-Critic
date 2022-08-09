import pickle


episode_file = 'test_trajectory.pkl'


def load_episode_to_buffer(buffer):
    with open(episode_file, 'rb') as f:
        trajectories = pickle.load(f)

    episode_step = len(trajectories)
    episode_reward = float(sum([t[3] for t in trajectories]))

    # get everything except info at the last
    trajectories = [t[:-1] for t in trajectories]
    buffer.extend(trajectories)

    return episode_step, episode_reward
