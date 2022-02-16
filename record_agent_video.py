from PIL import Image
from tqdm import trange

from common.carla_environment.environment import CarlaEnv
from gym.wrappers import RecordVideo
from gym.wrappers.monitoring.video_recorder import VideoRecorder


num_steps = 10000


if __name__ == '__main__':
    env = CarlaEnv()
    recorder = VideoRecorder(env, base_path='carla_agent_videos')

    env.reset()
    env.test_carla_agent(num_steps, recorder)

    recorder.close()
    env.close()
