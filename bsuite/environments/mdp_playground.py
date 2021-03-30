# pylint: disable=g-bad-file-header
# Copyright 2019 .... All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""The MDP Playground reinforcement learning environment."""

# Import all packages

from mdp_playground.envs import RLToyEnv #mdp_playground

# import collections
from bsuite.experiments.mdp_playground import sweep
from bsuite.environments import base
from bsuite.utils.gym_wrapper import DMEnvFromGym, space2spec
import dm_env
from dm_env import specs
from dm_env import StepType
import gym
import numpy as np

# def ohe_observation(obs):

class DM_RLToyEnv(base.Environment):
  """A wrapper to convert an RLToyEnv Gym environment from MDP Playground to a
  base.Environment which is a subclass of dm_env.Environment.
  Based on the DMEnvFromGym in gym_wrapper.py"""

  def __init__(self, max_episode_len=100, **config: dict):
    self.gym_env = gym.make("RLToy-v0", **config)
    self.dm_env = DMEnvFromGym(self.gym_env)

    self.max_episode_len = max_episode_len
    self._raw_return = 0.
    self._best_episode = 0.
    self._episode_return = 0.

    self.bsuite_num_episodes = sweep.NUM_EPISODES

    super(DM_RLToyEnv, self).__init__()
    # Convert gym action and observation spaces to dm_env specs.
    # self._observation_spec = space2spec(self.gym_env.observation_space,
    #                                     name='observations')
    # self._action_spec = space2spec(self.gym_env.action_space, name='actions')
    # self._reset_next_step = True

  def reset(self) -> dm_env.TimeStep:
    self._episode_return = 0.
    dm_env_reset = self.dm_env.reset()
    ohe_obs = np.zeros(shape=(self.gym_env.observation_space.n,), dtype=np.float32) #hack
    ohe_obs[dm_env_reset.observation] = 1
    # dm_env_reset.observation = ohe_obs
    return dm_env.restart(ohe_obs)

  def step(self, action: int) -> dm_env.TimeStep:
    dm_env_step = self.dm_env.step(action)
    
    #hack set reward as 0 if dm_env_step.reward returns None which happens in case of restart()
    self._raw_return += 0. if dm_env_step.reward is None else dm_env_step.reward
    self._episode_return += 0. if dm_env_step.reward is None else dm_env_step.reward

    if self.gym_env.total_transitions_episode > self.max_episode_len:
      self._best_episode = max(self._episode_return, self._best_episode)
      dm_env_step = dm_env.truncation(dm_env_step.reward, dm_env_step.observation)

    ohe_obs = np.zeros(shape=(self.gym_env.observation_space.n,), dtype=np.float32) #hack #TODO bsuite/baselines/tf/dqn agent doesn't allow discrete states
    ohe_obs[dm_env_step.observation] = 1
    # dm_env_step.observation = ohe_obs

    # return corresponding TimeStep object based on step_type
    if dm_env_step.step_type == StepType.FIRST:
      return dm_env.restart(ohe_obs)
    elif dm_env_step.step_type == StepType.LAST:
      return dm_env.termination(dm_env_step.reward, ohe_obs)
    else:
      return dm_env.transition(dm_env_step.reward, ohe_obs)

  def _step(self, action: int) -> dm_env.TimeStep:
    raise NotImplementedError('This environment implements its own auto-reset.')

  def _reset(self) -> dm_env.TimeStep:
    raise NotImplementedError('This environment implements its own auto-reset.')

  def close(self):
    self.gym_env.close()

  def observation_spec(self): ##TODO changed for OHE #hack
    return specs.BoundedArray(shape=(self.gym_env.observation_space.n,), dtype=np.float32, minimum=0.0,
                              maximum=1.0, name='observations')
    # return self.dm_env.observation_spec()

  def action_spec(self):
    return self.dm_env.action_spec()

  def bsuite_info(self):
    return dict(raw_return=self._raw_return,
                best_episode=self._best_episode)
