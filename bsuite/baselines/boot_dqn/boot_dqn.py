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
"""A simple implementation of Bootstrapped DQN with prior networks.

References:
1. https://arxiv.org/abs/1703.07608
2. https://arxiv.org/abs/1602.04621
3. https://arxiv.org/abs/1806.03335

This implementation is potentially inefficient, in that it does not parallelize
computation, but it is much more readable and clear than complex TF ops.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

import collections

from bsuite.baselines import base
from bsuite.baselines import replay
import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
import trfl
from typing import Callable, Sequence


class BootstrappedDqn(base.Agent):
  """Bootstrapped DQN with additive prior functions."""

  def __init__(
      self,
      obs_spec: dm_env.specs.Array,
      action_spec: dm_env.specs.BoundedArray,
      ensemble: Sequence[snt.AbstractModule],
      batch_size: int,
      agent_discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      optimizer: tf.train.Optimizer,
      mask_prob: float = 0.5,
      epsilon_fn: Callable[[int], float] = lambda _: 0.,
      seed: int = None,
  ):
    """Bootstrapped DQN with additive prior functions."""
    # Dqn configurations.
    self._ensemble = ensemble
    self._num_actions = action_spec.maximum - action_spec.minimum + 1
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._min_replay_size = min_replay_size
    self._epsilon_fn = epsilon_fn
    self._replay = replay.Replay(capacity=replay_capacity)
    self._mask_prob = mask_prob
    self._rng = np.random.RandomState(seed)
    tf.set_random_seed(seed)

    self._total_steps = 0
    self._total_episodes = 0
    self._active_head = 0
    self._num_ensemble = len(ensemble)

    # Making the tensorflow graph
    session = tf.Session()

    # Placeholders for tf graph = (obs, action, reward, discount, obs, mask)
    o_tm1 = tf.placeholder(shape=(None,) + obs_spec.shape, dtype=obs_spec.dtype)
    a_tm1 = tf.placeholder(shape=(None,), dtype=action_spec.dtype)
    r_t = tf.placeholder(shape=(None,), dtype=tf.float32)
    d_t = tf.placeholder(shape=(None,), dtype=tf.float32)
    o_t = tf.placeholder(shape=(None,) + obs_spec.shape, dtype=obs_spec.dtype)
    m_t = tf.placeholder(shape=(None, self._num_ensemble), dtype=tf.float32)

    losses = []
    value_fns = []
    for k, model in enumerate(self._ensemble):
      q_values = model(o_tm1)
      target_value = tf.reduce_max(model(o_t), axis=-1)

      train_value = trfl.batched_index(q_values, a_tm1)
      target_y = r_t + agent_discount * d_t * tf.stop_gradient(target_value)
      loss = tf.square(train_value - target_y) * m_t[:, k]
      value_fn = session.make_callable(q_values, [o_tm1])

      losses.append(loss)
      value_fns.append(value_fn)

    sgd_op = optimizer.minimize(tf.stack(losses))
    self._value_fns = value_fns
    self._sgd_step = session.make_callable(sgd_op,
                                           [o_tm1, a_tm1, r_t, d_t, o_t, m_t])
    session.run(tf.global_variables_initializer())

  def policy(self, timestep: dm_env.TimeStep) -> int:
    """Select actions according to epsilon-greedy policy."""
    if self._rng.rand() < self._epsilon_fn(self._total_steps):
      action = self._rng.randint(self._num_actions)
    else:
      batched_obs = np.expand_dims(timestep.observation, axis=0)
      q_values = self._value_fns[self._active_head](batched_obs)[0]
      action = np.argmax(q_values)
    return np.int32(action)

  def update(self,
             old_step: dm_env.TimeStep,
             action: int,
             new_step: dm_env.TimeStep):
    """Takes in a transition from the environment."""
    self._total_steps += 1
    if new_step.last():
      self._total_episodes += 1
      self._active_head = np.random.randint(self._num_ensemble)

    if self._is_time_to_sgd:
      minibatch = self._replay.sample(self._batch_size)
      self._sgd_step(*minibatch)

    if not old_step.last():
      self._replay.add(TransitionWithMask(
          o_tm1=old_step.observation,
          a_tm1=action,
          r_t=new_step.reward,
          d_t=new_step.discount,
          o_t=new_step.observation,
          m_t=self._rng.binomial(1, self._mask_prob, self._num_ensemble)
      ))

  @property
  def _is_time_to_sgd(self) -> bool:
    enough_replay = self._replay.size > self._min_replay_size
    is_update_step = self._total_steps % self._sgd_period == 0
    return enough_replay and is_update_step


TransitionWithMask = collections.namedtuple(
    'TransitionWithMask', ['o_tm1', 'a_tm1', 'r_t', 'd_t', 'o_t', 'm_t'])


class BatchFlattenMLP(snt.AbstractModule):
  """A simple multilayer perceptron which flattens all non-batch dimensions."""

  def __init__(self, output_sizes, name='simple_mlp'):
    self._output_sizes = output_sizes
    super(BatchFlattenMLP, self).__init__(name=name)

  def _build(self, inputs):
    inputs = snt.BatchFlatten()(inputs)
    outputs = snt.nets.MLP(self._output_sizes)(inputs)
    return outputs


class NetworkWithPrior(snt.AbstractModule):
  """Combine network with additive untrainable "prior network"."""

  def __init__(self,
               network: snt.AbstractModule,
               prior_network: snt.AbstractModule,
               prior_scale: float = 1.):
    super(NetworkWithPrior, self).__init__(name='network_with_prior')
    self._network = network
    self._prior_network = prior_network
    self._prior_scale = prior_scale

  def _build(self, inputs: tf.Tensor) -> tf.Tensor:
    q_values = self._network(inputs)
    prior_q_values = self._prior_network(inputs)
    return q_values + self._prior_scale * tf.stop_gradient(prior_q_values)


def make_ensemble(num_ensemble: int,
                  num_hidden_layers: int,
                  num_units: int,
                  num_actions: int,
                  prior_scale: float) -> Sequence[snt.AbstractModule]:
  """Convenience function to make an ensemble from flags."""
  output_sizes = [num_units] * num_hidden_layers + [num_actions]
  ensemble = []
  for _ in range(num_ensemble):
    network = BatchFlattenMLP(output_sizes)
    prior_network = BatchFlattenMLP(output_sizes)
    ensemble.append(NetworkWithPrior(network, prior_network, prior_scale))
  return ensemble


def default_agent(obs_spec: dm_env.specs.Array,
                  action_spec: dm_env.specs.DiscreteArray) -> BootstrappedDqn:
  """Initialize a Bootstrapped DQN agent with default parameters."""

  ensemble = make_ensemble(
      num_ensemble=16,
      num_hidden_layers=2,
      num_units=256,
      num_actions=action_spec.num_values,
      prior_scale=1.)

  return BootstrappedDqn(
      obs_spec=obs_spec,
      action_spec=action_spec,
      ensemble=ensemble,
      batch_size=32,
      agent_discount=.99,
      replay_capacity=16384,
      min_replay_size=128,
      sgd_period=16,
      optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
      epsilon_fn=lambda x: 0.05,
      seed=42)
