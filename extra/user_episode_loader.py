import os
import pickle


class UserEpisodeAdder:
    def __init__(self, n_episodes):
        self.n_episodes = n_episodes

        self.episode_file = 'user_episodes/user-episode-1.pkl'

        assert os.path.exists(self.episode_file), os.path.abspath(self.episode_file)

        # for straight road training
        self.episodes_chunk = [
            (0, 100),
        ]

        # for turning training
        # self.episodes_chunk = [
        #     (0, 100),
        #     (430, 530),
        #     (910, 1020),
        #     (1410, 1510)
        # ]

        self.add_every = self._calculate_adding_interval()

    def should_add_user_episode(self, current_episode):
        return current_episode % self.add_every == 0

    def get_episode(self, current_episode):
        full_episode = self._load_user_episode_from_file()
        chunk_index = max(0, int(current_episode / self.add_every) - 1)
        start, stop = self.episodes_chunk[chunk_index]

        chunk_episode = full_episode[start:stop]
        chunk_episode_reward = self._calculate_episode_reward(chunk_episode)

        return chunk_episode, len(chunk_episode), chunk_episode_reward

    def _calculate_adding_interval(self):
        return int(self.n_episodes / len(self.episodes_chunk))

    def _load_user_episode_from_file(self):
        with open(self.episode_file, 'rb') as f:
            return pickle.load(f)

    def _calculate_episode_reward(self, episode):
        return float(sum([t[3] for t in episode]))
