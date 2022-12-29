# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""MNIST classification as a bandit.

In this environment, we test the agent's generalization ability, and abstract
away exploration/planning/memory etc -- i.e. a bandit, with no 'state'.
"""

from typing import Optional

from bsuite.environments import base
from bsuite.experiments.mnist import sweep
from bsuite.utils import datasets

import dm_env
from dm_env import specs
import numpy as np


class MNISTBandit(base.Environment):
  """MNIST classification as a bandit environment."""

  def __init__(self, fraction: float = 1., seed: Optional[int] = None):
    """Loads the MNIST training set (60K images & labels) as numpy arrays.

    Args:
      fraction: What fraction of the training set to keep (default is all).
      seed: Optional integer. Seed for numpy's random number generator (RNG).
    """
    super().__init__()
    (images, labels), _ = datasets.load_mnist()

    num_data = len(labels)

    self._num_data = int(fraction * num_data)
    self._image_shape = images.shape[1:]

    self._images = images[:self._num_data]
    self._labels = labels[:self._num_data]
    self._rng = np.random.RandomState(seed)
    self._correct_label = None

    self._total_regret = 0.
    self._optimal_return = 1.

    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _reset(self) -> dm_env.TimeStep:
    """Agent gets an MNIST image to 'classify' using its next action."""
    idx = self._rng.randint(self._num_data)
    image = self._images[idx].astype(np.float32) / 255
    self._correct_label = self._labels[idx]

    return dm_env.restart(observation=image)

  def _step(self, action: int) -> dm_env.TimeStep:
    """+1/-1 for correct/incorrect guesses. This also terminates the episode."""
    correct = action == self._correct_label
    reward = 1. if correct else -1.
    self._total_regret += self._optimal_return - reward
    observation = np.zeros(shape=self._image_shape, dtype=np.float32)
    return dm_env.termination(reward=reward, observation=observation)

  def observation_spec(self):
    return specs.Array(
        shape=self._image_shape, dtype=np.float32, name='observation')

  def action_spec(self):
    return specs.DiscreteArray(num_values=10, name='action')

  def bsuite_info(self):
    return dict(total_regret=self._total_regret)
