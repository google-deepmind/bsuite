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
"""A simple TensorFlow-based implementation of a recurrent actor-critic.

References:
1. "Simple Statistical Gradient-Following Algorithms for Connectionist
    Reinforcement Learning" (Williams, 1992).
2. "Long Short-Term Memory" (Hochreiter, 1991).

Links:
1. http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf.
2. https://www.bioinf.jku.at/publications/older/2604.pdf
"""

from typing import Sequence

from bsuite.baselines import base
from bsuite.baselines.utils import sequence

import dm_env
from dm_env import specs
import sonnet as snt
import tensorflow as tf
import tree
import trfl


class ActorCriticRNN(base.Agent):
  """A recurrent TensorFlow-based feedforward actor-critic implementation."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.Array,
      network: 'PolicyValueRNN',
      optimizer: snt.Optimizer,
      max_sequence_length: int,
      td_lambda: float,
      discount: float,
      seed: int,
      entropy_cost: float = 0.,
  ):
    """A recurrent actor-critic agent."""

    # Internalise network and optimizer.
    self._forward = tf.function(network)
    self._network = network
    self._optimizer = optimizer

    # Initialise recurrent state.
    self._state = network.initial_state(1)
    self._rollout_initial_state = network.initial_state(1)

    # Set seed and internalise hyperparameters.
    tf.random.set_seed(seed)
    self._discount = discount
    self._td_lambda = td_lambda
    self._entropy_cost = entropy_cost

    # Initialise rolling experience buffer.
    self._buffer = sequence.Buffer(obs_spec, action_spec, max_sequence_length)

  @tf.function
  def _step(self, trajectory: sequence.Trajectory):
    """Do a batch of SGD on actor + critic loss on a sequence of experience."""
    observations, actions, rewards, discounts = trajectory

    # Add dummy batch dimensions.
    actions = tf.expand_dims(actions, axis=-1)  # [T, 1]
    rewards = tf.expand_dims(rewards, axis=-1)  # [T, 1]
    discounts = tf.expand_dims(discounts, axis=-1)  # [T, 1]
    observations = tf.expand_dims(observations, axis=1)  # [T+1, 1, ...]

    # Extract final observation for bootstrapping.
    observations, final_observation = observations[:-1], observations[-1]

    with tf.GradientTape() as tape:
      # Build actor and critic losses.
      (logits, values), state = snt.dynamic_unroll(
          self._network, observations, self._rollout_initial_state)
      (_, bootstrap_value), state = self._network(final_observation, state)
      values = tf.squeeze(values, axis=-1)
      bootstrap_value = tf.squeeze(bootstrap_value, axis=-1)
      critic_loss, (advantages, _) = trfl.td_lambda(
          state_values=values,
          rewards=rewards,
          pcontinues=self._discount * discounts,
          bootstrap_value=bootstrap_value,
          lambda_=self._td_lambda)
      actor_loss = trfl.discrete_policy_gradient_loss(
          logits, actions, advantages)
      entropy_loss = trfl.discrete_policy_entropy_loss(logits).loss
      loss = actor_loss + critic_loss + self._entropy_cost * entropy_loss
      loss = tf.reduce_mean(loss)

    gradients = tape.gradient(loss, self._network.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 5.)
    self._optimizer.apply(gradients, self._network.trainable_variables)

    return state

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    """Selects actions according to the latest softmax policy."""
    if timestep.first():
      self._state = self._network.initial_state(1)
      self._rollout_initial_state = self._network.initial_state(1)
    observation = tf.expand_dims(timestep.observation, axis=0)
    (logits, _), self._state = self._forward(observation, self._state)
    return tf.random.categorical(logits, num_samples=1).numpy().squeeze()

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
  ):
    """Receives a transition and performs a learning update."""
    self._buffer.append(timestep, action, new_timestep)

    if self._buffer.full() or new_timestep.last():
      trajectory = self._buffer.drain()
      trajectory = tree.map_structure(tf.convert_to_tensor, trajectory)
      self._rollout_initial_state = self._step(trajectory)


class PolicyValueRNN(snt.RNNCore):
  """A recurrent multi-layer perceptron with a value and a policy head."""

  def __init__(self, hidden_sizes: Sequence[int], num_actions: int):
    super().__init__(name='policy_value_net')
    self._torso = snt.nets.MLP(hidden_sizes, activate_final=True, name='torso')
    self._core = snt.LSTM(hidden_sizes[-1], name='rnn')
    self._policy_head = snt.Linear(num_actions, name='policy_head')
    self._value_head = snt.Linear(1, name='value_head')

  def __call__(self, inputs: tf.Tensor, state: snt.LSTMState):
    flat_inputs = snt.Flatten()(inputs)
    embedding = self._torso(flat_inputs)
    lstm_output, next_state = self._core(embedding, state)
    embedding += tf.nn.relu(lstm_output)  # Note: skip connection.
    logits = self._policy_head(embedding)
    value = self._value_head(embedding)
    return (logits, value), next_state

  def initial_state(self, *args, **kwargs) -> snt.LSTMState:
    """Creates the core initial state."""
    return self._core.initial_state(*args, **kwargs)


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray) -> base.Agent:
  """Initialize a DQN agent with default parameters."""
  network = PolicyValueRNN(
      hidden_sizes=[64, 64],
      num_actions=action_spec.num_values,
  )
  return ActorCriticRNN(
      obs_spec=obs_spec,
      action_spec=action_spec,
      network=network,
      optimizer=snt.optimizers.Adam(learning_rate=3e-3),
      max_sequence_length=32,
      td_lambda=0.9,
      discount=0.99,
      seed=42,
  )
