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

import dm_env
import numpy as np
import sonnet as snt
import tensorflow.compat.v2 as tf
import tree
import trfl


class ActorCriticRNN(base.Agent):
  """A recurrent TensorFlow-based feedforward actor-critic implementation."""

  def __init__(
      self,
      obs_spec: dm_env.specs.Array,
      network: snt.RNNCore,
      optimizer: snt.Optimizer,
      sequence_length: int,
      td_lambda: float,
      discount: float,
      seed: int,
  ):
    """A recurrent actor-critic agent."""

    # Internalise network and optimizer.
    self._forward = tf.function(network)
    self._network = network
    self._optimizer = optimizer

    # Initialise recurrent state.
    self._state: snt.LSTMState = network.initial_state(1)
    self._rollout_initial_state: snt.LSTMState = network.initial_state(1)

    # Set seed and internalise hyperparameters.
    tf.random.set_seed(seed)
    self._sequence_length = sequence_length
    self._num_transitions_in_buffer = 0
    self._discount = discount
    self._td_lambda = td_lambda

    # Initialise rolling experience buffer.
    shapes = [obs_spec.shape, (), (), (), ()]
    dtypes = [obs_spec.dtype, np.int32, np.float32, np.float32, np.float32]
    self._buffer = [
        np.zeros(shape=(self._sequence_length, 1) + shape, dtype=dtype)
        for shape, dtype in zip(shapes, dtypes)
    ]

  @tf.function
  def _step(self, sequence: Sequence[tf.Tensor]):
    """Do a batch of SGD on actor + critic loss on a sequence of experience."""
    (observations, actions, rewards, discounts, masks, final_obs,
     final_mask) = sequence
    masks = tf.expand_dims(masks, axis=-1)

    with tf.GradientTape() as tape:
      # Build actor and critic losses.
      state = self._rollout_initial_state
      logits_sequence = []
      values = []
      for t in range(self._sequence_length):
        (logits, value), state = self._network((observations[t], masks[t]),
                                               state)
        logits_sequence.append(logits)
        values.append(value)
      (_, bootstrap_value), _ = self._network((final_obs, final_mask), state)
      values = tf.squeeze(tf.stack(values, axis=0), axis=-1)
      logits = tf.stack(logits_sequence, axis=0)
      bootstrap_value = tf.squeeze(bootstrap_value, axis=-1)
      critic_loss, (advantages, _) = trfl.td_lambda(
          state_values=values,
          rewards=rewards,
          pcontinues=self._discount * discounts,
          bootstrap_value=bootstrap_value,
          lambda_=self._td_lambda)
      actor_loss = trfl.discrete_policy_gradient_loss(logits, actions,
                                                      advantages)
      loss = tf.reduce_mean(actor_loss + critic_loss)

    gradients = tape.gradient(loss, self._network.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 5.)
    self._optimizer.apply(gradients, self._network.trainable_variables)

    return state

  def policy(self, timestep: dm_env.TimeStep) -> base.Action:
    """Selects actions according to the latest softmax policy."""
    observation = tf.expand_dims(timestep.observation, axis=0)
    mask = tf.expand_dims(float(not timestep.first()), axis=0)
    (logits, _), self._state = self._forward((observation, mask), self._state)
    return tf.random.categorical(logits, num_samples=1).numpy().squeeze()

  def update(self, timestep: dm_env.TimeStep, action: base.Action,
             new_timestep: dm_env.TimeStep):
    """Receives a transition and performs a learning update."""

    # Insert this step into our rolling window 'batch'.
    items = [
        timestep.observation, action, new_timestep.reward,
        new_timestep.discount,
        float(not timestep.first())
    ]
    for buf, item in zip(self._buffer, items):
      buf[self._num_transitions_in_buffer % self._sequence_length, 0] = item
    self._num_transitions_in_buffer += 1

    # When the batch is full, do a step of SGD.
    if self._num_transitions_in_buffer % self._sequence_length != 0:
      return

    transitions = (
        self._buffer + [
            tf.expand_dims(new_timestep.observation, axis=0),  # final_obs
            tf.expand_dims(float(not new_timestep.first()),
                           axis=0),  # final_mask
        ])
    self._rollout_initial_state = self._step(transitions)


class PolicyValueRNN(snt.RNNCore):
  """A recurrent multilayer perceptron with a value and a policy head."""

  def __init__(self, hidden_sizes: Sequence[int], num_actions: int):
    super().__init__(name='policy_value_net')
    self._num_actions = num_actions
    self._torso = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP(hidden_sizes, activate_final=True, name='net'),
    ])
    self._core = snt.DeepRNN([
        snt.Flatten(),
        snt.LSTM(hidden_sizes[-1], name='rnn'),
    ])
    self._policy_head = snt.Linear(num_actions, name='policy')
    self._value_head = snt.Linear(1, name='value')

  def __call__(self, inputs_and_mask, state: snt.LSTMState):
    inputs, mask = inputs_and_mask
    state = tree.map_structure(lambda x: x * mask, state)
    embedding = self._torso(inputs)
    lstm_output, next_state = self._core(inputs, state)
    lstm_output = tf.nn.relu(lstm_output) + embedding  # 'skip connection'.
    logits = self._policy_head(lstm_output)
    value = self._value_head(lstm_output)
    return (logits, value), next_state

  def initial_state(self, *args, **kwargs):
    """Creates the core initial state."""
    return self._core.initial_state(*args, **kwargs)


def default_agent(obs_spec: dm_env.specs.Array,
                  action_spec: dm_env.specs.DiscreteArray):
  """Initialize a DQN agent with default parameters."""
  network = PolicyValueRNN(
      hidden_sizes=[64, 64, 32],
      num_actions=action_spec.num_values,
  )
  return ActorCriticRNN(
      obs_spec=obs_spec,
      network=network,
      optimizer=snt.optimizers.Adam(learning_rate=3e-3),
      sequence_length=32,
      td_lambda=0.9,
      discount=0.99,
      seed=42,
  )
