import os
import time
import pytz
from datetime import datetime
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.cuda.amp as amp
import tqdm
from setproctitle import setproctitle
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from common.utils import CHECKPOINT_FORMAT, normalize_image
from sac.model import RenderTester


DEFAULT_TIME_TO_SLEEP = 0.1

MEAN = np.tile([0.3171, 0.3183, 0.3779], 2)
STD = np.tile([0.1406, 0.0594, 0.0925], 2)


def train_loop(model, config, update_kwargs):
    with SummaryWriter(log_dir=os.path.join(config.log_dir, 'trainer'), comment='trainer') as writer:
        n_initial_samples = model.collector.n_total_steps
        n_initial_episodes = model.collector.n_episodes
        while model.collector.n_total_steps == n_initial_samples:
            time.sleep(0.1)

        time_to_sleep = DEFAULT_TIME_TO_SLEEP
        setproctitle(title='trainer')
        for epoch in range(config.initial_epoch + 1, config.n_epochs + 1):
            epoch_critic_loss = 0.0
            epoch_actor_loss = 0.0
            epoch_alpha = 0.0
            mean_episode_reward = 0.0
            mean_episode_steps = 0.0
            with tqdm.trange(config.n_updates, desc=f'Training {epoch}/{config.n_epochs}') as pbar:
                for i in pbar:
                    info = model.update(**update_kwargs, nth_update=(i+1))

                    n_samples = model.collector.n_total_steps
                    n_episodes = model.collector.n_episodes
                    buffer_size = model.replay_buffer.size
                    try:
                        update_sample_ratio = (config.n_samples_per_update * model.global_step) / \
                                              (n_samples - n_initial_samples)
                    except ZeroDivisionError:
                        update_sample_ratio = config.update_sample_ratio
                    recent_slice = slice(max(n_episodes - 100, n_initial_episodes + 1), n_episodes)
                    mean_episode_reward = np.mean(model.collector.episode_rewards[recent_slice])
                    mean_episode_steps = np.mean(model.collector.episode_steps[recent_slice])
                    epoch_critic_loss += (info['critic_loss'] - epoch_critic_loss) / (i + 1)
                    if 'actor_loss' in info:
                        epoch_actor_loss += (info['actor_loss'] - epoch_actor_loss) / ((i + 1) / config.actor_update_frequency)
                    epoch_alpha += (info['temperature_parameter'] - epoch_alpha) / (i + 1)
                    for item in info:
                        writer.add_scalar(tag=f'train/{item}', scalar_value=info[item],
                                          global_step=model.global_step)
                    writer.add_scalar(tag='train/mean_episode_reward', scalar_value=mean_episode_reward,
                                      global_step=model.global_step)
                    writer.add_scalar(tag='train/mean_episode_steps', scalar_value=mean_episode_steps,
                                      global_step=model.global_step)
                    writer.add_scalar(tag='train/buffer_size', scalar_value=buffer_size,
                                      global_step=model.global_step)
                    writer.add_scalar(tag='train/update_sample_ratio', scalar_value=update_sample_ratio,
                                      global_step=model.global_step)
                    pbar.set_postfix(OrderedDict([('global_step', model.global_step),
                                                  ('episode_reward', mean_episode_reward),
                                                  ('episode_steps', mean_episode_steps),
                                                  ('n_samples', f'{n_samples:.2E}'),
                                                  ('update/sample', f'{update_sample_ratio:.1f}')]))
                    if update_sample_ratio < config.update_sample_ratio:
                        model.collector.pause()
                        time_to_sleep = DEFAULT_TIME_TO_SLEEP
                    else:
                        model.collector.resume()
                        if update_sample_ratio > config.update_sample_ratio * 1.1:
                            time.sleep(time_to_sleep)
                            time_to_sleep += DEFAULT_TIME_TO_SLEEP

            writer.add_scalar(tag='epoch/critic_loss', scalar_value=epoch_critic_loss, global_step=epoch)
            writer.add_scalar(tag='epoch/actor_loss', scalar_value=epoch_actor_loss, global_step=epoch)
            writer.add_scalar(tag='epoch/temperature_parameter', scalar_value=epoch_alpha, global_step=epoch)
            writer.add_scalar(tag='epoch/mean_episode_reward', scalar_value=mean_episode_reward, global_step=epoch)
            writer.add_scalar(tag='epoch/mean_episode_steps', scalar_value=mean_episode_steps, global_step=epoch)

            writer.flush()
            model.save_model(path=os.path.join(config.checkpoint_dir, 'latest.pkl'))
            if epoch % config.checkpoint_save_frequency == 0:
                checkpoint_name = CHECKPOINT_FORMAT(epoch=epoch, reward=mean_episode_reward)
                model.save_model(path=os.path.join(config.checkpoint_dir, checkpoint_name))

            if not model.collector.are_samplers_running:
                print('all samplers stopped, terminate training process')
                break


def train(model, config):
    update_kwargs = config.build_from_keys(['batch_size',
                                            'normalize_rewards',
                                            'reward_scale',
                                            'adaptive_entropy',
                                            'clip_gradient',
                                            'gamma',
                                            'soft_tau'])
    if config.RNN_encoder:
        update_kwargs.update(step_size=config.step_size)

    if config.target_entropy is None:
        update_kwargs.update(target_entropy=-1.0 * config.action_dim)
    else:
        assert isinstance(config.target_entropy, float)
        update_kwargs.update(target_entropy=config.target_entropy)

    print(f'Start parallel sampling using {config.n_samplers} samplers '
          f'at {tuple(map(str, model.collector.devices))}.')

    model.collector.eval()
    while model.replay_buffer.size < 10 * config.n_samples_per_update:
        model.sample(n_episodes=10 * int(np.ceil(config.n_samples_per_update / config.max_episode_steps)),
                     max_episode_steps=config.max_episode_steps,
                     deterministic=False,
                     random_sample=True,
                     render=config.render)

    model.collector.train()
    model.async_sample(n_episodes=np.inf,
                       deterministic=False,
                       random_sample=False,
                       **config.build_from_keys(['max_episode_steps',
                                                 'render',
                                                 'log_episode_video',
                                                 'log_dir']))

    try:
        train_loop(model, config, update_kwargs)
    except KeyboardInterrupt:
        model.collector.terminate()
    except Exception:
        raise


def test(model, config):
    with SummaryWriter(log_dir=config.log_dir) as writer:
        print(f'Start parallel sampling using {config.n_samplers} samplers '
              f'at {tuple(map(str, model.collector.devices))}.')

        model.sample(random_sample=False,
                     **config.build_from_keys([
                         'n_episodes',
                         'max_episode_steps',
                         'deterministic',
                         'render',
                         'log_episode_video',
                         'log_dir'
                     ]))

        episode_steps = np.asanyarray(model.collector.episode_steps)
        episode_rewards = np.asanyarray(model.collector.episode_rewards)
        average_reward = episode_rewards / episode_steps
        writer.add_histogram(tag='test/cumulative_reward', values=episode_rewards)
        writer.add_histogram(tag='test/average_reward', values=average_reward)
        writer.add_histogram(tag='test/episode_steps', values=episode_steps)

        results = {
            'Metrics': ['Cumulative Reward', 'Average Reward', 'Episode Steps'],
            'Mean': list(map(np.mean, [episode_rewards, average_reward, episode_steps])),
            'Stddev': list(map(np.std, [episode_rewards, average_reward, episode_steps])),
        }
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
        except ImportError:
            for metric, mean, stddev in zip(results['Metrics'], results['Mean'], results['Stddev']):
                print(f'{metric}: {dict(mean=mean, stddev=stddev)}')


def test_render(model: RenderTester, config):
    model.state_encoder.reset()
    observation = model.env.reset()
    extra_state = None
    if hasattr(model.env, 'first_extra_state'):
        extra_state = model.env.first_extra_state

    rewards = 0

    for step in trange(1, config.max_episode_steps + 1):
        normalized_obs = normalize_image(observation, MEAN, STD).transpose((2, 0, 1))
        extra_state_tensor = torch.tensor(extra_state, dtype=torch.float32, device=model.model_device)
        state = model.state_encoder.encode(normalized_obs, extra_state_tensor, return_tensor=True)

        action = model.actor.get_action(state, deterministic=True)

        next_observation, reward, done, info = model.env.step(action)

        if 'extra_state' in info:
            next_extra_state = info['extra_state']

        should_stop = info.get('should_stop', False)

        rewards += reward
        observation = next_observation
        extra_state = next_extra_state

        if done or should_stop:
            print(f'done: {done}, should_stop: {should_stop}')
            break

    model.env.close()
    print(f'accumulate reward is {rewards}')
    
    bkk_tz = pytz.timezone('Asia/Bangkok')
    now = datetime.now(bkk_tz)
    now_str = now.strftime('%Y-%m-%dT%H-%M-%S')
    model.env.plot_control_graph(os.path.join(config.video_dir, f'command_epoch_{config.initial_epoch}_{now_str}.jpeg'))
    plt.clf()
    model.env.plot_speed_graph(os.path.join(config.video_dir, f'speed_epoch_{config.initial_epoch}_{now_str}.jpeg'))
    plt.clf()
    model.env.plot_distance_graph(os.path.join(config.video_dir, f'distance_epoch_{config.initial_epoch}_{now_str}.jpeg'))
