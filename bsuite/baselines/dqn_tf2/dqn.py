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
"""A simple TensorFlow 2-based DQN implementation.

Reference: "Playing atari with deep reinforcement learning" (Mnih et al, 2015).
Link: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf.
"""

# Import all packages

from bsuite.baselines import base
from bsuite.baselines.utils import replay

import dm_env
from dm_env import specs

import numpy as np
import sonnet as snt
import tensorflow.compat.v2 as tf
from trfl.action_value_ops import qlearning


class DQNTF2(base.Agent):
  """A simple DQN agent using TF2."""

  def __init__(
      self,
      action_spec: specs.DiscreteArray,
      online_network: snt.Module,
      target_network: snt.Module,
      batch_size: int,
      discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      target_update_period: int,
      optimizer: snt.Optimizer,
      epsilon: float,
      seed: int = None,
  ):

    # DQN configuration and hyperparameters.
    self._num_actions = action_spec.num_values
    self._discount = discount
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._optimizer = optimizer
    self._epsilon = epsilon
    self._total_steps = 0
    self._replay = replay.Replay(capacity=replay_capacity)
    self._min_replay_size = min_replay_size

    tf.random.set_seed(seed)
    self._rng = np.random.RandomState(seed)

    # Internalize the networks.
    self._online_network = online_network
    self._target_network = target_network
    self._forward = tf.function(online_network)

  def policy(self, timestep: dm_env.TimeStep) -> base.Action:
    # Epsilon-greedy policy.
    if self._rng.rand() < self._epsilon:
      return np.random.randint(self._num_actions)
    q_values = self._forward(timestep.observation[None, ...])
    return int(np.argmax(q_values))

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
  ):
    # Add this transition to replay.
    self._replay.add([
        timestep.observation,
        action,
        new_timestep.reward,
        new_timestep.discount,
        new_timestep.observation,
    ])

    self._total_steps += 1
    if self._total_steps % self._sgd_period != 0:
      return

    if self._replay.size < self._min_replay_size:
      return

    # Do a batch of SGD.
    transitions = self._replay.sample(self._batch_size)
    self._training_step(transitions)

    # Periodically update target network variables.
    if self._total_steps % self._target_update_period == 0:
      for target, param in zip(self._target_network.trainable_variables,
                               self._online_network.trainable_variables):
        target.assign(param)

  @tf.function
  def _training_step(self, transitions):
    with tf.GradientTape() as tape:
      o_tm1, a_tm1, r_t, d_t, o_t = transitions
      r_t = tf.cast(r_t, tf.float32)
      d_t = tf.cast(d_t, tf.float32)
      q_tm1 = self._online_network(o_tm1)
      q_t = self._target_network(o_t)

      loss = qlearning(q_tm1, a_tm1, r_t, d_t * self._discount, q_t).loss

    params = self._online_network.trainable_variables
    grads = tape.gradient(loss, params)
    self._optimizer.apply(grads, params)
    return loss


def default_agent(obs_spec: specs.Array, action_spec: specs.DiscreteArray):
  """Initialize a DQN agent with default parameters."""
  del obs_spec  # Unused.
  hidden_units = [50, 50]
  online_network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP(hidden_units + [action_spec.num_values]),
  ])
  target_network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP(hidden_units + [action_spec.num_values]),
  ])
  return DQNTF2(
      action_spec=action_spec,
      online_network=online_network,
      target_network=target_network,
      batch_size=32,
      discount=0.99,
      replay_capacity=10000,
      min_replay_size=100,
      sgd_period=1,
      target_update_period=4,
      optimizer=snt.optimizers.Adam(learning_rate=1e-3),
      epsilon=0.05,
      seed=42)
