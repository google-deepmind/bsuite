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
"""A simple implementation of Bootstrapped DQN with prior networks.

References:
1. "Deep Exploration via Bootstrapped DQN" (Osband et al., 2016)
2. "Deep Exploration via Randomized Value Functions" (Osband et al., 2017)
3. "Randomized Prior Functions for Deep RL" (Osband et al, 2018)

Links:
1. https://arxiv.org/abs/1602.04621
2. https://arxiv.org/abs/1703.07608
3. https://arxiv.org/abs/1806.03335

Notes:

- This agent is implemented with TensorFlow 2 and Sonnet 2. For installation
  instructions for these libraries, see the README.md in the parent folder.
- This implementation is potentially inefficient, as it does not parallelise
  computation across the ensemble for simplicity and readability.
"""

import copy
from typing import Callable, NamedTuple, Optional, Sequence

from bsuite.baselines import base
from bsuite.baselines.utils import replay

import dm_env
from dm_env import specs
import numpy as np
import sonnet as snt
import tensorflow as tf
import tree


class BootstrappedDqn(base.Agent):
  """Bootstrapped DQN with additive prior functions."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      ensemble: Sequence[snt.Module],
      batch_size: int,
      discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      target_update_period: int,
      optimizer: snt.Optimizer,
      mask_prob: float,
      noise_scale: float,
      epsilon_fn: Callable[[int], float] = lambda _: 0.,
      seed: Optional[int] = None,
  ):
    """Bootstrapped DQN with additive prior functions."""
    # Agent components.
    self._ensemble = ensemble
    self._forward = [tf.function(net) for net in ensemble]
    self._target_ensemble = [copy.deepcopy(network) for network in ensemble]
    self._num_ensemble = len(ensemble)
    self._optimizer = optimizer
    self._replay = replay.Replay(capacity=replay_capacity)

    # Create variables for each network in the ensemble
    for network in ensemble:
      snt.build(network, (None, *obs_spec.shape))

    # Agent hyperparameters.
    self._num_actions = action_spec.num_values
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._min_replay_size = min_replay_size
    self._epsilon_fn = epsilon_fn
    self._mask_prob = mask_prob
    self._noise_scale = noise_scale
    self._rng = np.random.RandomState(seed)
    self._discount = discount

    # Agent state.
    self._total_steps = tf.Variable(1)
    self._active_head = 0
    tf.random.set_seed(seed)

  @tf.function
  def _step(self, transitions: Sequence[tf.Tensor]):
    """Does a step of SGD for the whole ensemble over `transitions`."""
    o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions
    variables = tree.flatten(
        [model.trainable_variables for model in self._ensemble])
    with tf.GradientTape() as tape:
      losses = []
      for k in range(self._num_ensemble):
        net = self._ensemble[k]
        target_net = self._target_ensemble[k]

        # Q-learning loss with added reward noise + half-in bootstrap.
        q_values = net(o_tm1)
        one_hot_actions = tf.one_hot(a_tm1, depth=self._num_actions)
        train_value = tf.reduce_sum(q_values * one_hot_actions, axis=-1)
        target_value = tf.stop_gradient(tf.reduce_max(target_net(o_t), axis=-1))
        target_y = r_t + z_t[:, k] + self._discount * d_t * target_value
        loss = tf.square(train_value - target_y) * m_t[:, k]
        losses.append(loss)

      loss = tf.reduce_mean(tf.stack(losses))
      gradients = tape.gradient(loss, variables)
    self._total_steps.assign_add(1)
    self._optimizer.apply(gradients, variables)

    # Periodically update the target network.
    if tf.math.mod(self._total_steps, self._target_update_period) == 0:
      for k in range(self._num_ensemble):
        for src, dest in zip(self._ensemble[k].variables,
                             self._target_ensemble[k].variables):
          dest.assign(src)

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    """Select values via Thompson sampling, then use epsilon-greedy policy."""
    if self._rng.rand() < self._epsilon_fn(self._total_steps.numpy()):
      return self._rng.randint(self._num_actions)

    # Greedy policy, breaking ties uniformly at random.
    batched_obs = tf.expand_dims(timestep.observation, axis=0)
    q_values = self._forward[self._active_head](batched_obs)[0].numpy()
    action = self._rng.choice(np.flatnonzero(q_values == q_values.max()))
    return int(action)

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
  ):
    """Update the agent: add transition to replay and periodically do SGD."""
    if new_timestep.last():
      self._active_head = self._rng.randint(self._num_ensemble)

    self._replay.add(
        TransitionWithMaskAndNoise(
            o_tm1=timestep.observation,
            a_tm1=action,
            r_t=np.float32(new_timestep.reward),
            d_t=np.float32(new_timestep.discount),
            o_t=new_timestep.observation,
            m_t=self._rng.binomial(1, self._mask_prob,
                                   self._num_ensemble).astype(np.float32),
            z_t=self._rng.randn(self._num_ensemble).astype(np.float32) *
            self._noise_scale,
        ))

    if self._replay.size < self._min_replay_size:
      return

    if tf.math.mod(self._total_steps, self._sgd_period) == 0:
      minibatch = self._replay.sample(self._batch_size)
      minibatch = [tf.convert_to_tensor(x) for x in minibatch]
      self._step(minibatch)


class TransitionWithMaskAndNoise(NamedTuple):
  o_tm1: np.ndarray
  a_tm1: base.Action
  r_t: float
  d_t: float
  o_t: np.ndarray
  m_t: np.ndarray
  z_t: np.ndarray


class NetworkWithPrior(snt.Module):
  """Combines network with additive untrainable "prior network"."""

  def __init__(self,
               network: snt.Module,
               prior_network: snt.Module,
               prior_scale: float = 1.):
    super().__init__(name='network_with_prior')
    self._network = network
    self._prior_network = prior_network
    self._prior_scale = prior_scale

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    q_values = self._network(inputs)
    prior_q_values = self._prior_network(inputs)
    return q_values + self._prior_scale * tf.stop_gradient(prior_q_values)


def make_ensemble(num_actions: int,
                  num_ensemble: int = 20,
                  num_hidden_layers: int = 2,
                  num_units: int = 50,
                  prior_scale: float = 3.) -> Sequence[snt.Module]:
  """Convenience function to make an ensemble from flags."""
  output_sizes = [num_units] * num_hidden_layers + [num_actions]
  ensemble = []
  for _ in range(num_ensemble):
    network = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP(output_sizes),
    ])
    prior_network = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP(output_sizes),
    ])
    ensemble.append(NetworkWithPrior(network, prior_network, prior_scale))
  return ensemble


def default_agent(
    obs_spec: specs.Array,
    action_spec: specs.DiscreteArray,
    num_ensemble: int = 20,
) -> BootstrappedDqn:
  """Initialize a Bootstrapped DQN agent with default parameters."""
  ensemble = make_ensemble(
      num_actions=action_spec.num_values, num_ensemble=num_ensemble)
  optimizer = snt.optimizers.Adam(learning_rate=1e-3)
  return BootstrappedDqn(
      obs_spec=obs_spec,
      action_spec=action_spec,
      ensemble=ensemble,
      batch_size=128,
      discount=.99,
      replay_capacity=10000,
      min_replay_size=128,
      sgd_period=1,
      target_update_period=4,
      optimizer=optimizer,
      mask_prob=0.5,
      noise_scale=0.0,
      epsilon_fn=lambda t: 10 / (10 + t),
      seed=42,
  )
