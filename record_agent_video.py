from PIL import Image
from tqdm import trange

from common.carla_environment.environment import CarlaEnv
from gym.wrappers import RecordVideo
from gym.wrappers.monitoring.video_recorder import VideoRecorder


num_steps = 200


if __name__ == '__main__':
    # env = CarlaEnv()
    env = CarlaEnv(run_backward=True)
    recorder = VideoRecorder(env, base_path='carla_agent_videos')
    start_step = 0

    completed_lap = False
    while not completed_lap:
        env.reset()
        completed_lap = env.test_carla_agent(num_steps, start_step=start_step)

        start_step = start_step + num_steps + 1

    recorder.close()
    env.close()
