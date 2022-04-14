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

import copy
from typing import Optional, Sequence

from bsuite.baselines import base
from bsuite.baselines.utils import replay

import dm_env
from dm_env import specs
import numpy as np
import sonnet as snt
import tensorflow as tf


class DQN(base.Agent):
  """A simple DQN agent using TF2."""

  def __init__(
      self,
      action_spec: specs.DiscreteArray,
      network: snt.Module,
      batch_size: int,
      discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      target_update_period: int,
      optimizer: snt.Optimizer,
      epsilon: float,
      seed: Optional[int] = None,
  ):

    # Internalise hyperparameters.
    self._num_actions = action_spec.num_values
    self._discount = discount
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._epsilon = epsilon
    self._min_replay_size = min_replay_size

    # Seed the RNG.
    tf.random.set_seed(seed)
    self._rng = np.random.RandomState(seed)

    # Internalise the components (networks, optimizer, replay buffer).
    self._optimizer = optimizer
    self._replay = replay.Replay(capacity=replay_capacity)
    self._online_network = network
    self._target_network = copy.deepcopy(network)
    self._forward = tf.function(network)
    self._total_steps = tf.Variable(0)

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    # Epsilon-greedy policy.
    if self._rng.rand() < self._epsilon:
      return self._rng.randint(self._num_actions)

    observation = tf.convert_to_tensor(timestep.observation[None, ...])
    # Greedy policy, breaking ties uniformly at random.
    q_values = self._forward(observation).numpy()
    action = self._rng.choice(np.flatnonzero(q_values == q_values.max()))
    return int(action)

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

    self._total_steps.assign_add(1)
    if tf.math.mod(self._total_steps, self._sgd_period) != 0:
      return

    if self._replay.size < self._min_replay_size:
      return

    # Do a batch of SGD.
    transitions = self._replay.sample(self._batch_size)
    self._training_step(transitions)

  @tf.function
  def _training_step(self, transitions: Sequence[tf.Tensor]) -> tf.Tensor:
    """Does a step of SGD on a batch of transitions."""
    o_tm1, a_tm1, r_t, d_t, o_t = transitions
    r_t = tf.cast(r_t, tf.float32)  # [B]
    d_t = tf.cast(d_t, tf.float32)  # [B]
    o_tm1 = tf.convert_to_tensor(o_tm1)
    o_t = tf.convert_to_tensor(o_t)

    with tf.GradientTape() as tape:
      q_tm1 = self._online_network(o_tm1)  # [B, A]
      q_t = self._target_network(o_t)  # [B, A]

      onehot_actions = tf.one_hot(a_tm1, depth=self._num_actions)  # [B, A]
      qa_tm1 = tf.reduce_sum(q_tm1 * onehot_actions, axis=-1)  # [B]
      qa_t = tf.reduce_max(q_t, axis=-1)  # [B]

      # One-step Q-learning loss.
      target = r_t + d_t * self._discount * qa_t
      td_error = qa_tm1 - target
      loss = 0.5 * tf.reduce_mean(td_error**2)  # []

    # Update the online network via SGD.
    variables = self._online_network.trainable_variables
    gradients = tape.gradient(loss, variables)
    self._optimizer.apply(gradients, variables)

    # Periodically copy online -> target network variables.
    if tf.math.mod(self._total_steps, self._target_update_period) == 0:
      for target, param in zip(self._target_network.trainable_variables,
                               self._online_network.trainable_variables):
        target.assign(param)
    return loss


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray):
  """Initialize a DQN agent with default parameters."""
  del obs_spec  # Unused.
  network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([50, 50, action_spec.num_values]),
  ])
  optimizer = snt.optimizers.Adam(learning_rate=1e-3)
  return DQN(
      action_spec=action_spec,
      network=network,
      batch_size=32,
      discount=0.99,
      replay_capacity=10000,
      min_replay_size=100,
      sgd_period=1,
      target_update_period=4,
      optimizer=optimizer,
      epsilon=0.05,
      seed=42)
