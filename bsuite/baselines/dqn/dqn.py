# pylint: disable=g-bad-file-header
# Copyright 2019 The bsuite Authors. All Rights Reserved.
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
"""A simple TensorFlow-based DQN implementation.

Reference: "Playing atari with deep reinforcement learning" (Mnih et al, 2015).
Link: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.baselines import base
from bsuite.baselines import replay

import dm_env
from dm_env import specs
import numpy as np
import sonnet as snt
import tensorflow as tf
import trfl


class DQN(base.Agent):
  """A simple TensorFlow-based DQN implementation."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.BoundedArray,
      online_network: snt.AbstractModule,
      target_network: snt.AbstractModule,
      batch_size: int,
      reward_bound: float,
      agent_discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      target_update_period: int,
      optimizer: tf.train.Optimizer,
      epsilon: float,
      seed: int = None,
  ):
    """A simple DQN agent."""

    # DQN configuration and hyperparameters.
    self._action_spec = action_spec
    self._num_actions = action_spec.maximum - action_spec.minimum + 1
    self._reward_bound = reward_bound
    self._agent_discount = agent_discount
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._optimizer = optimizer
    self._epsilon = epsilon
    self._total_steps = 0
    self._replay = replay.Replay(capacity=replay_capacity)
    self._min_replay_size = min_replay_size
    tf.set_random_seed(seed)
    self._rng = np.random.RandomState(seed)

    # Make the TensorFlow graph.
    o = tf.placeholder(shape=obs_spec.shape, dtype=obs_spec.dtype)
    q = online_network(tf.expand_dims(o, 0))

    o_tm1 = tf.placeholder(shape=(None,) + obs_spec.shape, dtype=obs_spec.dtype)
    a_tm1 = tf.placeholder(shape=(None,), dtype=action_spec.dtype)
    r_t = tf.placeholder(shape=(None,), dtype=tf.float32)
    d_t = tf.placeholder(shape=(None,), dtype=tf.float32)
    o_t = tf.placeholder(shape=(None,) + obs_spec.shape, dtype=obs_spec.dtype)

    q_tm1 = online_network(o_tm1)
    q_t = target_network(o_t)
    loss = trfl.qlearning(q_tm1, a_tm1, r_t, agent_discount * d_t, q_t).loss

    sgd_op = self._optimizer.minimize(loss)
    periodic_target_update_op = trfl.periodic_target_update(
        target_variables=target_network.variables,
        source_variables=online_network.variables,
        update_period=target_update_period)

    with tf.control_dependencies([sgd_op, periodic_target_update_op]):
      loss = tf.reduce_mean(loss)

    # Make session and callables.
    session = tf.Session()
    self._sgd_fn = session.make_callable(loss, [o_tm1, a_tm1, r_t, d_t, o_t])
    self._value_fn = session.make_callable(q, [o])
    session.run(tf.global_variables_initializer())

  def policy(self, timestep: dm_env.TimeStep) -> base.Action:
    """Select actions according to epsilon-greedy policy."""
    if self._rng.rand() < self._epsilon:
      return self._rng.randint(self._num_actions)

    q_values = self._value_fn(timestep.observation)
    return np.argmax(q_values)

  def update(self, old_step: dm_env.TimeStep, action: base.Action,
             new_step: dm_env.TimeStep):
    """Takes in a transition from the environment."""

    # Do reward clipping.
    new_step = new_step._replace(
        reward=np.clip(new_step.reward, -self._reward_bound,
                       self._reward_bound))

    # Add this transition to replay.
    self._replay.add([
        old_step.observation,
        action,
        new_step.reward,
        new_step.discount,
        new_step.observation,
    ])

    self._total_steps += 1
    if self._total_steps % self._sgd_period != 0:
      return

    if self._replay.size < self._min_replay_size:
      return

    # Do a batch of SGD.
    minibatch = self._replay.sample(self._batch_size)
    self._sgd_fn(*minibatch)
