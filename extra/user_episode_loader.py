import os
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


class UserEpisodeAdder:
    def __init__(self, n_episodes):
        self.n_episodes = n_episodes

        self.episode_file = 'user_episodes/usable_test_drive1.pkl'

        assert os.path.exists(episode_file)

        self.episodes_chunk = [
            (0, 100),
            (400, 530),
            (900, 1100),
            (1320, 1500)
        ]

        self.add_every = self._calculate_adding_interval()

    def should_add_user_episode(self, current_episode):
        return current_episode % self.add_every == 0

    def get_episode(self, current_episode):
        full_episode = self._load_user_episode_from_file()
        chunk_index = int(current_episode / self.add_every)
        start, stop = self.episodes_chunk[chunk_index]

        chunk_episode = full_episode[start, stop]
        chunk_episode_reward = self._calculate_episode_reward(chunk_episode)

        return chunk_episode, len(chunk_episode), chunk_episode_reward

    def _calculate_adding_interval(self):
        return int(self.n_episodes / len(self.episodes_chunk))

    def _load_user_episode_from_file(self):
        with open(self.episode_file, 'rb') as f:
            return pickle.load(f)

    def _calculate_episode_reward(self, episode):
        return float(sum([t[3] for t in episode]))
