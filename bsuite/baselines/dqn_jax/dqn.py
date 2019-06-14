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
"""A simple JAX-based DQN implementation.

Reference: "Playing atari with deep reinforcement learning" (Mnih et al, 2015).
Link: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf.
"""

# Import all packages

from bsuite.baselines import base
from bsuite.baselines.utils import replay

import dm_env
from dm_env import specs

import jax
from jax import lax
from jax import numpy as jnp
from jax import random

from jax.experimental import optimizers
from jax.experimental import stax

import numpy as np
from typing import Any, Callable, Sequence

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class DQNJAX(base.Agent):
  """A simple DQN agent using JAX."""

  def __init__(
      self,
      action_spec: specs.DiscreteArray,
      network: Network,
      parameters: NetworkParameters,
      batch_size: int,
      discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      target_update_period: int,
      learning_rate: float,
      epsilon: float,
      seed: int = None,
  ):

    # DQN configuration and hyperparameters.
    self._num_actions = action_spec.num_values
    self._discount = discount
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._epsilon = epsilon
    self._total_steps = 0
    self._replay = replay.Replay(capacity=replay_capacity)
    self._min_replay_size = min_replay_size

    self._rng = np.random.RandomState(seed)

    def loss(online_params, target_params, transitions):
      o_tm1, a_tm1, r_t, d_t, o_t = transitions
      q_tm1 = network(online_params, o_tm1)
      q_t = network(target_params, o_t)
      q_target = r_t + d_t * discount * jnp.max(q_t, axis=-1)
      q_a_tm1 = jax.vmap(lambda q, a: q[a])(q_tm1, a_tm1)
      td_error = q_a_tm1 - lax.stop_gradient(q_target)

      return jnp.mean(td_error**2)

    # Internalize the networks.
    self._network = network
    self._parameters = parameters
    self._target_parameters = parameters

    # This function computes dL/dTheta
    self._grad = jax.jit(jax.grad(loss))
    self._forward = jax.jit(network)

    # Make an Adam optimizer.
    opt_init, opt_update, get_params = optimizers.adam(step_size=learning_rate)
    self._opt_update = jax.jit(opt_update)
    self._opt_state = opt_init(parameters)
    self._get_params = get_params

  def policy(self, timestep: dm_env.TimeStep) -> base.Action:
    # Epsilon-greedy policy.
    if np.random.rand() < self._epsilon:
      return np.random.randint(self._num_actions)
    q_values = self._forward(self._parameters, timestep.observation[None, ...])
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
    gradient = self._grad(self._parameters, self._target_parameters,
                          transitions)
    self._opt_state = self._opt_update(self._total_steps, gradient,
                                       self._opt_state)
    self._parameters = self._get_params(self._opt_state)

    # Periodically update target network variables.
    if self._total_steps % self._target_update_period == 0:
      self._target_parameters = self._parameters


def default_agent(obs_spec: specs.Array, action_spec: specs.DiscreteArray):
  """Initialize a DQN agent with default parameters."""
  network_init, network = stax.serial(
      stax.Flatten,
      stax.Dense(50),
      stax.Relu,
      stax.Dense(50),
      stax.Relu,
      stax.Dense(action_spec.num_values),
  )
  _, network_params = network_init(
      random.PRNGKey(seed=1), (-1,) + obs_spec.shape)

  return DQNJAX(
      action_spec=action_spec,
      network=network,
      parameters=network_params,
      batch_size=32,
      discount=0.99,
      replay_capacity=10000,
      min_replay_size=100,
      sgd_period=1,
      target_update_period=4,
      learning_rate=1e-3,
      epsilon=0.05,
      seed=42)
