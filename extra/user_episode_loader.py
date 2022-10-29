import os
import pickle


# user_episodes/user-episode-1.pkl has no turning reward
# user_episodes/user-episode-4.pkl has turning reward and no steering penalty


class UserEpisodeAdder:
    def __init__(self, n_episodes):
        # in training, n_episodes is infinity, set it to arbitrary value
        if n_episodes == float("inf"):
            self.n_episodes = 1000
        else:
            self.n_episodes = n_episodes

        self.episode_file = 'user_episodes/user-episode-5fps-1.pkl'

        assert os.path.exists(self.episode_file), os.path.abspath(self.episode_file)

        # for straight road training
        self.episodes_chunk = [
            (0, 100),
        ]

        # for turning training of user-episode-1.pkl
        # self.episodes_chunk = [
        #     (0, 100),
        #     (430, 530),
        #     (910, 1020),
        #     (1410, 1510)
        # ]

        # for turning training of user-episode-2.pkl and episode user-episode-3.pkl
        # self.episode_chunks = [
        #     (0, 100),
        #     (430, 530),
        #     (930, 1030),
        #     (1420, 1500)
        # ]

        # for episode user-episode-4.pkl and episode user-episode-5.pkl
        # self.episode_chunks = [
        #     (0, 100),
        #     (405, 480),
        #     (865, 935),
        #     (1305, 1390)
        # ]

        self.add_every = self._calculate_adding_interval()

    def should_add_user_episode(self, current_episode):
        return current_episode % self.add_every == 0

    def get_episode(self, current_episode):
        full_episode = self._load_user_episode_from_file()
        chunk_index = max(0, int(current_episode / self.add_every) - 1)
        start, stop = self.episode_chunks[chunk_index]

        chunk_episode = full_episode[start:stop]
        chunk_episode_reward = self._calculate_episode_reward(chunk_episode)

        return chunk_episode, len(chunk_episode), chunk_episode_reward

    def get_all_episodes(self):
        full_episode = self._load_user_episode_from_file()

        episode_chunks = []

        for start, stop in self.episode_chunks:
            chunk_episode = full_episode[start:stop]
            chunk_episode_reward = self._calculate_episode_reward(chunk_episode)

            episode_chunks.append((chunk_episode, len(chunk_episode), chunk_episode_reward))

        return episode_chunks

    def _calculate_adding_interval(self):
        return int(self.n_episodes / len(self.episode_chunks))

    def _load_user_episode_from_file(self):
        with open(self.episode_file, 'rb') as f:
            return pickle.load(f)

    def _calculate_episode_reward(self, episode):
        return float(sum([t[3] for t in episode]))
