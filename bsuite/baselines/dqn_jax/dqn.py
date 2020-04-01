# python3
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

from typing import Any, Callable, NamedTuple, Sequence

from bsuite.baselines import base
from bsuite.baselines.utils import replay

import dm_env
from dm_env import specs
import haiku as hk
import jax
from jax import lax
from jax.experimental import optix
import jax.numpy as jnp
import numpy as np
import rlax

QNetwork = Callable[[jnp.ndarray], jnp.ndarray]  # observations -> action values


class TrainingState(NamedTuple):
  """Holds the agent's training state."""
  params: hk.Params
  target_params: hk.Params
  opt_state: Any
  step: int


class DQN(base.Agent):
  """A simple DQN agent using JAX."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      network: QNetwork,
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

    # Store hyperparameters.
    self._num_actions = action_spec.num_values
    self._discount = discount
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._epsilon = epsilon
    self._total_steps = 0
    self._replay = replay.Replay(capacity=replay_capacity)
    self._min_replay_size = min_replay_size

    # Internalize the networks.
    rng = hk.PRNGSequence(seed)
    init, forward = hk.transform(network)
    dummy_obs = np.zeros((1, *obs_spec.shape), obs_spec.dtype)
    initial_params = init(next(rng), dummy_obs)
    initial_target_params = init(next(rng), dummy_obs)

    # Make an Adam optimizer.
    opt_init, opt_update = optix.adam(learning_rate)
    initial_opt_state = opt_init(initial_params)

    # This carries the agent state relevant to training.
    self._state = TrainingState(
        params=initial_params,
        target_params=initial_target_params,
        opt_state=initial_opt_state,
        step=0)

    def loss(params: hk.Params,
             target_params: hk.Params,
             transitions: Sequence[jnp.ndarray]) -> jnp.ndarray:
      """Computes the standard TD(0) Q-learning loss on batch of transitions."""
      o_tm1, a_tm1, r_t, d_t, o_t = transitions
      q_tm1 = forward(params, o_tm1)
      q_t = forward(target_params, o_t)
      td_error = jax.vmap(rlax.q_learning)(q_tm1, a_tm1, r_t, d_t, q_t)
      return jnp.mean(td_error**2)

    def update(state: TrainingState,
               transitions: Sequence[jnp.ndarray]) -> TrainingState:
      """Performs a batch of SGD."""
      gradients = jax.grad(loss)(state.params, state.target_params, transitions)
      updates, new_opt_state = opt_update(gradients, state.opt_state)
      new_params = optix.apply_updates(state.params, updates)

      # Periodically update the target network parameters.
      target_params = lax.cond(
          pred=jnp.mod(state.step, target_update_period) == 0,
          true_operand=None,
          true_fun=lambda _: new_params,
          false_operand=None,
          false_fun=lambda _: state.target_params)

      return TrainingState(
          params=new_params,
          target_params=target_params,
          opt_state=new_opt_state,
          step=state.step + 1)

    self._update = jax.jit(update)
    self._forward = jax.jit(forward)

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    """Selects actions according to an epsilon-greedy policy."""
    if np.random.rand() < self._epsilon:
      return np.random.randint(self._num_actions)

    observation = timestep.observation[None, ...]
    q_values = self._forward(self._state.params, observation)
    action = int(np.argmax(q_values))
    return action

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
  ):
    """Adds transition to replay and periodically does SGD."""
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
    self._state = self._update(self._state, transitions)


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray) -> base.Agent:
  """Initialize a DQN agent with default parameters."""

  def network(inputs: jnp.ndarray) -> jnp.ndarray:
    flat_inputs = hk.Flatten()(inputs)
    mlp = hk.nets.MLP([64, 64, action_spec.num_values])
    action_values = mlp(flat_inputs)
    return action_values

  return DQN(
      obs_spec=obs_spec,
      action_spec=action_spec,
      network=network,
      batch_size=32,
      discount=0.99,
      replay_capacity=10000,
      min_replay_size=100,
      sgd_period=1,
      target_update_period=4,
      learning_rate=1e-3,
      epsilon=0.05,
      seed=42)
