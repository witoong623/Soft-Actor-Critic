from PIL import Image
from tqdm import trange

from common.carla_environment.environment import CarlaEnv

num_episodes = 200
num_steps = 1000
num_image = 1

def save_image(img):
    global num_image

    pil_image = Image.fromarray(img)
    pil_image.save(f'carla_images/carla_image_{num_image}.jpeg')
    num_image += 1


if __name__ == '__main__':
    env = CarlaEnv()

    for _ in trange(num_episodes):
        env.reset()
        env.collect_env_images(num_steps, save_image)
