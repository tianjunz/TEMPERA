import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.wrappers.clip_action import ClipAction
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnvWrapper)
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

from data_utils import custom_load_dataset
from prompt_env import LMForwardEnvNoPrefix
from utils import *

try:
    import dmc2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

def make_env(seed, params, max_steps, num_processes, obs_size):
    def _thunk():
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = custom_load_dataset(params)
        # Set seed
        np.random.seed(params['seed'])
        import copy
        few_shot_train_sentences = []
        few_shot_train_labels = []
        number_dict = {x:0 for x in params['label_dict'].keys()}
        hundred_train_sentences, hundred_train_labels = random_sampling(all_train_sentences, all_train_labels, 100)
        for train_sentence, train_label in zip(hundred_train_sentences, hundred_train_labels):
            if number_dict[train_label] < int(params['example_pool_size']/len(number_dict.values())):
                few_shot_train_sentences.append(copy.deepcopy(train_sentence))
                few_shot_train_labels.append(copy.deepcopy(train_label))
                number_dict[train_label] += 1
            if sum(number_dict.values()) == params['example_pool_size']:
                break
        train_sentences, train_labels = few_shot_train_sentences, few_shot_train_labels

        raw_train_sentences, raw_train_labels = train_sentences[:params['num_shots']], train_labels[:params['num_shots']]
        raw_pool_sentences, raw_pool_labels = train_sentences[params['num_shots']:], train_labels[params['num_shots']:]
        if len(all_train_sentences) > 100:
            raw_all_train_sentences, raw_all_train_labels = random_sampling(all_train_sentences, all_train_labels, 100)
            if params['sub_sample']:
                import copy
                few_shot_train_sentences = []
                few_shot_train_labels = []
                number_dict = {x:0 for x in params['label_dict'].keys()}
                hundred_train_sentences, hundred_train_labels = random_sampling(all_train_sentences, all_train_labels, 1000)
                for train_sentence, train_label in zip(hundred_train_sentences, hundred_train_labels):
                    if number_dict[train_label] < 16:
                        few_shot_train_sentences.append(copy.deepcopy(train_sentence))
                        few_shot_train_labels.append(copy.deepcopy(train_label))
                        number_dict[train_label] += 1
                    if sum(number_dict.values()) == 16 * len(number_dict.values()):
                        break
                all_train_sentences, all_train_labels = few_shot_train_sentences, few_shot_train_labels        
        else:
            raw_all_train_sentences, raw_all_train_labels = all_train_sentences, all_train_labels

        if params['env_name'] == 'lmnoprefix':
            env = LMForwardEnvNoPrefix(params, raw_train_sentences, raw_train_labels, raw_all_train_sentences, raw_all_train_labels, raw_pool_sentences, raw_pool_labels, all_train_sentences, all_train_labels, max_steps, num_processes, obs_size, entropy_coef=params['entropy_coef'], loss_type=params['rew_type'], verbalizer=params['verbalizer'])
        print('Finish Build Environment')
 
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(seed,
        params, 
        max_steps, 
        num_processes,
        gamma, 
        obs_size,
        device):

    envs = [
        make_env(seed, params, max_steps, num_processes, obs_size)
    ]

    envs = DummyVecEnv1(envs, num_processes)
    envs = VecPyTorch(envs, device)

    return envs

def make_env_fseval(seed, params, max_steps, num_processes, obs_size):
    def _thunk():
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = custom_load_dataset(params)
        # Set seed
        np.random.seed(params['seed'])
        import copy
        few_shot_train_sentences = []
        few_shot_train_labels = []
        number_dict = {x:0 for x in params['label_dict'].keys()}
        hundred_train_sentences, hundred_train_labels = random_sampling(all_train_sentences, all_train_labels, 100)
        for train_sentence, train_label in zip(hundred_train_sentences, hundred_train_labels):
            if number_dict[train_label] < int(params['example_pool_size']/len(number_dict.values())):
                few_shot_train_sentences.append(copy.deepcopy(train_sentence))
                few_shot_train_labels.append(copy.deepcopy(train_label))
                number_dict[train_label] += 1
            if sum(number_dict.values()) == params['example_pool_size']:
                break
        train_sentences, train_labels = few_shot_train_sentences, few_shot_train_labels

        raw_train_sentences, raw_train_labels = train_sentences[:params['num_shots']], train_labels[:params['num_shots']]
        raw_pool_sentences, raw_pool_labels = train_sentences[params['num_shots']:], train_labels[params['num_shots']:]
        if len(all_train_sentences) > 100:
            raw_all_train_sentences, raw_all_train_labels = random_sampling(all_train_sentences, all_train_labels, 100)
        else:
            raw_all_train_sentences, raw_all_train_labels = all_train_sentences, all_train_labels
        np.random.seed(params['seed']+1)
        all_train_sentences, all_train_labels = random_sampling(all_train_sentences, all_train_labels, 16)
        
        if params['env_name'] == 'lmnoprefix':
            env = LMForwardEnvNoPrefix(params, raw_train_sentences, raw_train_labels, raw_all_train_sentences, raw_all_train_labels, raw_pool_sentences, raw_pool_labels, all_train_sentences, all_train_labels, max_steps, num_processes, obs_size, entropy_coef=0.0, loss_type=params['rew_type'], verbalizer=params['verbalizer'], evaluate=True)
        print('Finish Build Environment')

       
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs_fseval(seed,
        params, 
        max_steps, 
        num_processes,
        gamma, 
        obs_size,
        device):

    envs = [
        make_env_fseval(seed, params, max_steps, num_processes, obs_size)
    ]

    envs = DummyVecEnv1(envs, num_processes)
    envs = VecPyTorch(envs, device)

    return envs

def make_env_eval(seed, params, max_steps, num_processes, obs_size, change_params, i, gpu_id):
    def _thunk():
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = custom_load_dataset(params, change_params)
        # Set seed
        np.random.seed(params['seed'])
        import copy
        few_shot_train_sentences = []
        few_shot_train_labels = []
        number_dict = {x:0 for x in params['label_dict'].keys()}
        hundred_train_sentences, hundred_train_labels = random_sampling(all_train_sentences, all_train_labels, 100)
        for train_sentence, train_label in zip(hundred_train_sentences, hundred_train_labels):
            if number_dict[train_label] < int(params['example_pool_size']/len(number_dict.values())):
                few_shot_train_sentences.append(copy.deepcopy(train_sentence))
                few_shot_train_labels.append(copy.deepcopy(train_label))
                number_dict[train_label] += 1
            if sum(number_dict.values()) == params['example_pool_size']:
                break
        train_sentences, train_labels = few_shot_train_sentences, few_shot_train_labels

        raw_train_sentences, raw_train_labels = train_sentences[:params['num_shots']], train_labels[:params['num_shots']]
        raw_pool_sentences, raw_pool_labels = train_sentences[params['num_shots']:], train_labels[params['num_shots']:]
        if len(all_train_sentences) > 100:
            raw_all_train_sentences, raw_all_train_labels = random_sampling(all_train_sentences, all_train_labels, 100)
        else:
            raw_all_train_sentences, raw_all_train_labels = all_train_sentences, all_train_labels
        num_examples = int(len(all_test_sentences)/params['num_actors']) - 1
        test_senntences, test_labels = all_test_sentences[i*num_examples:(i+1)*num_examples], all_test_labels[i*num_examples:(i+1)*num_examples]

        if params['env_name'] == 'lmnoprefix':
            env = LMForwardEnvNoPrefix(params, raw_train_sentences, raw_train_labels, raw_all_train_sentences, raw_all_train_labels, raw_pool_sentences, raw_pool_labels, test_senntences, test_labels, max_steps, num_processes, obs_size, gpu_id, entropy_coef=0.0, loss_type=params['rew_type'], verbalizer=params['verbalizer'], evaluate=True)
        print('Environment actor ', i, ' on gpu ', gpu_id, flush=True)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs_eval(seed,
        params, 
        max_steps, 
        num_processes,
        gamma, 
        obs_size,
        change_params,
        i,
        gpu_id=0):

    envs = [
        make_env_eval(seed, params, max_steps, num_processes, obs_size, change_params, i, gpu_id)
    ]

    envs = DummyVecEnv1(envs, num_processes)

    _, _, all_test_sentences, all_test_labels = custom_load_dataset(params, change_params)
    num_examples = int(len(all_test_sentences)/params['num_actors']) - 1

    return envs, num_examples

def get_num_test(seed,
        params, 
        max_steps, 
        num_processes,
        gamma, 
        obs_size,
        i,
        gpu_id=0):
        
    _, _, all_test_sentences, all_test_labels = custom_load_dataset(params)
    num_examples = int(len(all_test_sentences)/params['num_actors']) - 1

    return num_examples

# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        if device == -1:
            self.device = 'cpu'
        else:
            self.device = 'cuda:'+str(device)
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        # print(obs.shape, reward.shape, done.shape)
        return obs, reward, done, info

class TestVecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(TestVecPyTorch, self).__init__(venv)
        self.device = 'cpu'
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.tensor(obs)
        return obs

    def step_async(self, actions):
        # if isinstance(actions, torch.LongTensor):
        #     # Squeeze the dimension for discrete actions
        #     actions = actions.squeeze(1)
        # actions = actions.squeeze(1)
        # actions = actions.numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.tensor(obs)
        reward = torch.tensor(reward)
        # print(obs.shape, reward.shape, done.shape)
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) /
                          np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs, self.clip_obs)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Type, Union
import numpy as np
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn

class DummyVecEnv1(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.
    :param env_fns: a list of functions
        that return environments to vectorize
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], num_processes):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((num_processes,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((num_processes,), dtype=bool)
        self.buf_rews = np.zeros((num_processes,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews, self.buf_dones, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions
            )
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                # self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        seeds = []
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[np.ndarray]:
        return [env.render(mode="rgb_array") for env in self.envs]

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.
        Otherwise (if ``self.num_envs == 1``), we pass the render call directly to the
        underlying environment.
        Therefore, some arguments such as ``mode`` will have values that are valid
        only when ``num_envs == 1``.
        :param mode: The rendering type.
        """
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                # self.buf_obs[key][env_idx] = obs
                self.buf_obs[key] = obs
            else:
                # self.buf_obs[key][env_idx] = obs[key]
                self.buf_obs[key] = obs[key]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(low=low,
                                           high=high,
                                           dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
