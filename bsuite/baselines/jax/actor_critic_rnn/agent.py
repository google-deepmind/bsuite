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
"""A simple recurrent actor-critic agent implemented in JAX + Haiku."""

from typing import Any, Callable, NamedTuple, Tuple

from bsuite.baselines import base
from bsuite.baselines.utils import sequence

import dm_env
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax

Logits = jnp.ndarray
Value = jnp.ndarray
LSTMState = Any
RecurrentPolicyValueNet = Callable[[jnp.ndarray, LSTMState],
                                   Tuple[Tuple[Logits, Value], LSTMState]]


class AgentState(NamedTuple):
  """Holds the network parameters, optimizer state, and RNN state."""
  params: hk.Params
  opt_state: Any
  rnn_state: LSTMState
  rnn_unroll_state: LSTMState


class ActorCriticRNN(base.Agent):
  """Recurrent actor-critic agent."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      network: RecurrentPolicyValueNet,
      initial_rnn_state: LSTMState,
      optimizer: optax.GradientTransformation,
      rng: hk.PRNGSequence,
      sequence_length: int,
      discount: float,
      td_lambda: float,
      entropy_cost: float = 0.,
  ):

    # Define loss function.
    def loss(trajectory: sequence.Trajectory, rnn_unroll_state: LSTMState):
      """"Actor-critic loss."""
      (logits, values), new_rnn_unroll_state = hk.dynamic_unroll(
          network, trajectory.observations[:, None, ...], rnn_unroll_state)
      seq_len = trajectory.actions.shape[0]
      td_errors = rlax.td_lambda(
          v_tm1=values[:-1, 0],
          r_t=trajectory.rewards,
          discount_t=trajectory.discounts * discount,
          v_t=values[1:, 0],
          lambda_=jnp.array(td_lambda),
      )
      critic_loss = jnp.mean(td_errors**2)
      actor_loss = rlax.policy_gradient_loss(
          logits_t=logits[:-1, 0],
          a_t=trajectory.actions,
          adv_t=td_errors,
          w_t=jnp.ones(seq_len))
      entropy_loss = jnp.mean(
          rlax.entropy_loss(logits[:-1, 0], jnp.ones(seq_len)))

      combined_loss = actor_loss + critic_loss + entropy_cost * entropy_loss

      return combined_loss, new_rnn_unroll_state

    # Transform the loss into a pure function.
    loss_fn = hk.without_apply_rng(hk.transform(loss)).apply

    # Define update function.
    @jax.jit
    def sgd_step(state: AgentState,
                 trajectory: sequence.Trajectory) -> AgentState:
      """Does a step of SGD over a trajectory."""
      gradients, new_rnn_state = jax.grad(
          loss_fn, has_aux=True)(state.params, trajectory,
                                 state.rnn_unroll_state)
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)
      return state._replace(
          params=new_params,
          opt_state=new_opt_state,
          rnn_unroll_state=new_rnn_state)

    # Initialize network parameters and optimiser state.
    init, forward = hk.without_apply_rng(hk.transform(network))
    dummy_observation = jnp.zeros((1, *obs_spec.shape), dtype=obs_spec.dtype)
    initial_params = init(next(rng), dummy_observation, initial_rnn_state)
    initial_opt_state = optimizer.init(initial_params)

    # Internalize state.
    self._state = AgentState(initial_params, initial_opt_state,
                             initial_rnn_state, initial_rnn_state)
    self._forward = jax.jit(forward)
    self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length)
    self._sgd_step = sgd_step
    self._rng = rng
    self._initial_rnn_state = initial_rnn_state

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    """Selects actions according to a softmax policy."""
    key = next(self._rng)
    observation = timestep.observation[None, ...]
    (logits, _), rnn_state = self._forward(self._state.params, observation,
                                           self._state.rnn_state)
    self._state = self._state._replace(rnn_state=rnn_state)
    action = jax.random.categorical(key, logits).squeeze()
    return int(action)

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
  ):
    """Adds a transition to the trajectory buffer and periodically does SGD."""
    if new_timestep.last():
      self._state = self._state._replace(rnn_state=self._initial_rnn_state)
    self._buffer.append(timestep, action, new_timestep)
    if self._buffer.full() or new_timestep.last():
      trajectory = self._buffer.drain()
      self._state = self._sgd_step(self._state, trajectory)


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray,
                  seed: int = 0) -> base.Agent:
  """Creates an actor-critic agent with default hyperparameters."""

  hidden_size = 256
  initial_rnn_state = hk.LSTMState(
      hidden=jnp.zeros((1, hidden_size), dtype=jnp.float32),
      cell=jnp.zeros((1, hidden_size), dtype=jnp.float32))

  def network(inputs: jnp.ndarray,
              state) -> Tuple[Tuple[Logits, Value], LSTMState]:
    flat_inputs = hk.Flatten()(inputs)
    torso = hk.nets.MLP([hidden_size, hidden_size])
    lstm = hk.LSTM(hidden_size)
    policy_head = hk.Linear(action_spec.num_values)
    value_head = hk.Linear(1)

    embedding = torso(flat_inputs)
    embedding, state = lstm(embedding, state)
    logits = policy_head(embedding)
    value = value_head(embedding)
    return (logits, jnp.squeeze(value, axis=-1)), state

  return ActorCriticRNN(
      obs_spec=obs_spec,
      action_spec=action_spec,
      network=network,
      initial_rnn_state=initial_rnn_state,
      optimizer=optax.adam(3e-3),
      rng=hk.PRNGSequence(seed),
      sequence_length=32,
      discount=0.99,
      td_lambda=0.9,
  )
