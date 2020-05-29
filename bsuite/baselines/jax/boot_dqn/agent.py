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


class TrainingState(NamedTuple):
  params: hk.Params
  target_params: hk.Params
  opt_state: Any
  step: int


class BootstrappedDqn(base.Agent):
  """Bootstrapped DQN with randomized prior functions."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      network: Callable[[jnp.ndarray], jnp.ndarray],
      num_ensemble: int,
      batch_size: int,
      discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      target_update_period: int,
      optimizer: optix.InitUpdate,
      mask_prob: float,
      noise_scale: float,
      epsilon_fn: Callable[[int], float] = lambda _: 0.,
      seed: int = 1,
  ):
    # Transform the (impure) network into a pure function.
    network = hk.transform(network)

    # Define loss function, including bootstrap mask `m_t` & reward noise `z_t`.
    def loss(params: hk.Params, target_params: hk.Params,
             transitions: Sequence[jnp.ndarray]) -> jnp.ndarray:
      """Q-learning loss with added reward noise + half-in bootstrap."""
      o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions
      q_tm1 = network.apply(params, o_tm1)
      q_t = network.apply(target_params, o_t)
      r_t += noise_scale * z_t
      batch_q_learning = jax.vmap(rlax.q_learning)
      td_error = batch_q_learning(q_tm1, a_tm1, r_t, discount * d_t, q_t)
      return jnp.mean(m_t * td_error**2)

    # Define update function for each member of ensemble..
    @jax.jit
    def sgd_step(state: TrainingState,
                 transitions: Sequence[jnp.ndarray]) -> TrainingState:
      """Does a step of SGD for the whole ensemble over `transitions`."""

      gradients = jax.grad(loss)(state.params, state.target_params, transitions)
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optix.apply_updates(state.params, updates)

      return TrainingState(
          params=new_params,
          target_params=state.target_params,
          opt_state=new_opt_state,
          step=state.step + 1)

    # Initialize parameters and optimizer state for an ensemble of Q-networks.
    rng = hk.PRNGSequence(seed)
    dummy_obs = np.zeros((1, *obs_spec.shape), jnp.float32)
    initial_params = [
        network.init(next(rng), dummy_obs) for _ in range(num_ensemble)
    ]
    initial_target_params = [
        network.init(next(rng), dummy_obs) for _ in range(num_ensemble)
    ]
    initial_opt_state = [optimizer.init(p) for p in initial_params]

    # Internalize state.
    self._ensemble = [
        TrainingState(p, tp, o, step=0) for p, tp, o in zip(
            initial_params, initial_target_params, initial_opt_state)
    ]
    self._forward = jax.jit(network.apply)
    self._sgd_step = sgd_step
    self._num_ensemble = num_ensemble
    self._optimizer = optimizer
    self._replay = replay.Replay(capacity=replay_capacity)

    # Agent hyperparameters.
    self._num_actions = action_spec.num_values
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._min_replay_size = min_replay_size
    self._epsilon_fn = epsilon_fn
    self._mask_prob = mask_prob

    # Agent state.
    self._active_head = self._ensemble[0]
    self._total_steps = 0

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    """Select values via Thompson sampling, then use epsilon-greedy policy."""
    self._total_steps += 1
    if np.random.rand() < self._epsilon_fn(self._total_steps):
      return np.random.randint(self._num_actions)

    # Greedy policy, breaking ties uniformly at random.
    batched_obs = timestep.observation[None, ...]
    q_values = self._forward(self._active_head.params, batched_obs)
    action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
    return int(action)

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
  ):
    """Update the agent: add transition to replay and periodically do SGD."""

    # Thompson sampling: every episode pick a new Q-network as the policy.
    if new_timestep.last():
      k = np.random.randint(self._num_ensemble)
      self._active_head = self._ensemble[k]

    # Generate bootstrapping mask & reward noise.
    mask = np.random.binomial(1, self._mask_prob, self._num_ensemble)
    noise = np.random.randn(self._num_ensemble)

    # Make transition and add to replay.
    transition = [
        timestep.observation,
        action,
        np.float32(new_timestep.reward),
        np.float32(new_timestep.discount),
        new_timestep.observation,
        mask,
        noise,
    ]
    self._replay.add(transition)

    if self._replay.size < self._min_replay_size:
      return

    # Periodically sample from replay and do SGD for the whole ensemble.
    if self._total_steps % self._sgd_period == 0:
      transitions = self._replay.sample(self._batch_size)
      o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions
      for k, state in enumerate(self._ensemble):
        transitions = [o_tm1, a_tm1, r_t, d_t, o_t, m_t[:, k], z_t[:, k]]
        self._ensemble[k] = self._sgd_step(state, transitions)

    # Periodically update target parameters.
    for k, state in enumerate(self._ensemble):
      if state.step % self._target_update_period == 0:
        self._ensemble[k] = state._replace(target_params=state.params)


def default_agent(
    obs_spec: specs.Array,
    action_spec: specs.DiscreteArray,
    seed: int = 0,
    num_ensemble: int = 20,
) -> BootstrappedDqn:
  """Initialize a Bootstrapped DQN agent with default parameters."""

  # Define network.
  prior_scale = 3.
  hidden_sizes = [50, 50]

  def network(inputs: jnp.ndarray) -> jnp.ndarray:
    """Simple Q-network with randomized prior function."""
    net = hk.nets.MLP([*hidden_sizes, action_spec.num_values])
    prior_net = hk.nets.MLP([*hidden_sizes, action_spec.num_values])
    x = hk.Flatten()(inputs)
    return net(x) + prior_scale * lax.stop_gradient(prior_net(x))

  optimizer = optix.adam(learning_rate=1e-3)
  return BootstrappedDqn(
      obs_spec=obs_spec,
      action_spec=action_spec,
      network=network,
      batch_size=128,
      discount=.99,
      num_ensemble=num_ensemble,
      replay_capacity=10000,
      min_replay_size=128,
      sgd_period=1,
      target_update_period=4,
      optimizer=optimizer,
      mask_prob=0.5,
      noise_scale=0.,
      epsilon_fn=lambda _: 0.,
      seed=seed,
  )
