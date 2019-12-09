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

# Import all packages

from bsuite.baselines import base
import dm_env
from dm_env import specs
import numpy as np
import sonnet as snt
import tensorflow as tf
import tree
from trfl.discrete_policy_gradient_ops import discrete_policy_gradient_loss
from trfl.value_ops import td_lambda as td_lambda_loss
from typing import Sequence


class ActorCriticRNN(base.Agent):
  """A recurrent TensorFlow-based feedforward actor-critic implementation."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      network: snt.RNNCore,
      optimizer: tf.train.Optimizer,
      sequence_length: int,
      td_lambda: float,
      agent_discount: float,
      seed: int,
  ):
    """A recurrent actor-critic agent."""
    del action_spec  # unused
    tf.set_random_seed(seed)
    self._sequence_length = sequence_length
    self._num_transitions_in_buffer = 0

    # Create the policy ops.
    obs = tf.placeholder(shape=(1,) + obs_spec.shape, dtype=obs_spec.dtype)
    mask = tf.placeholder(shape=(1,), dtype=tf.float32)
    state = self._placeholders_like(network.initial_state(batch_size=1))
    (online_logits, _), next_state = network((obs, mask), state)
    action = tf.squeeze(tf.multinomial(online_logits, 1, output_dtype=tf.int32))

    # Create placeholders and numpy arrays for learning from trajectories.
    shapes = [obs_spec.shape, (), (), (), ()]
    dtypes = [obs_spec.dtype, np.int32, np.float32, np.float32, np.float32]

    placeholders = [
        tf.placeholder(shape=(self._sequence_length, 1) + shape, dtype=dtype)
        for shape, dtype in zip(shapes, dtypes)]
    observations, actions, rewards, discounts, masks = placeholders

    # Build actor and critic losses.
    (logits, values), final_state = tf.nn.dynamic_rnn(
        network, (observations, tf.expand_dims(masks, -1)),
        initial_state=state, dtype=tf.float32, time_major=True)
    (_, bootstrap_value), _ = network((obs, mask), final_state)
    values, bootstrap_value = tree.map_structure(
        lambda t: tf.squeeze(t, axis=-1), (values, bootstrap_value))
    critic_loss, (advantages, _) = td_lambda_loss(
        state_values=values,
        rewards=rewards,
        pcontinues=agent_discount * discounts,
        bootstrap_value=bootstrap_value,
        lambda_=td_lambda)
    actor_loss = discrete_policy_gradient_loss(logits, actions, advantages)

    # Updates.
    grads_and_vars = optimizer.compute_gradients(actor_loss + critic_loss)
    grads, _ = tf.clip_by_global_norm([g for g, _ in grads_and_vars], 5.)
    grads_and_vars = [(g, pair[1]) for g, pair in zip(grads, grads_and_vars)]
    train_op = optimizer.apply_gradients(grads_and_vars)

    # Create TF session and callables.
    session = tf.Session()
    self._reset_fn = session.make_callable(
        network.initial_state(batch_size=1))
    self._policy_fn = session.make_callable(
        [action, next_state], [obs, mask, state])
    self._update_fn = session.make_callable(
        [train_op, final_state], placeholders + [obs, mask, state])
    session.run(tf.global_variables_initializer())

    # Initialize numpy buffers
    self.state = self._reset_fn()
    self.update_init_state = self._reset_fn()
    self.arrays = [
        np.zeros(shape=(self._sequence_length, 1) + shape, dtype=dtype)
        for shape, dtype in zip(shapes, dtypes)]

  def _placeholders_like(self, tensor_nest):
    """Create placeholders with given nested structure, shape, and type."""
    return tree.map_structure(
        lambda t: tf.placeholder(shape=t.shape, dtype=t.dtype), tensor_nest)

  def _compute_entropy_loss(self, logits):
    """Compute entropy loss for a set of policy logits."""
    policy = tf.nn.softmax(logits)
    log_policy = tf.nn.log_softmax(logits)
    return -tf.reduce_sum(-policy * log_policy)

  def policy(self, timestep: dm_env.TimeStep) -> base.Action:
    """Selects actions according to the latest softmax policy."""
    action, self.state = self._policy_fn(
        np.expand_dims(timestep.observation, 0),
        np.expand_dims(float(not timestep.first()), 0),
        self.state)
    return np.int32(action)

  def update(self, old_step: dm_env.TimeStep, action: base.Action,
             new_step: dm_env.TimeStep):
    """Receives a transition and performs a learning update."""

    # Insert this step into our rolling window 'batch'.
    items = [
        old_step.observation, action, new_step.reward, new_step.discount,
        float(not old_step.step_type.first())]
    for buf, item in zip(self.arrays, items):
      buf[self._num_transitions_in_buffer % self._sequence_length, 0] = item
    self._num_transitions_in_buffer += 1

    # When the batch is full, do a step of SGD.
    if self._num_transitions_in_buffer % self._sequence_length == 0:
      _, self.update_init_state = self._update_fn(*(self.arrays + [
          np.expand_dims(new_step.observation, 0),
          np.expand_dims(float(not new_step.step_type.first()), 0),
          self.update_init_state]))


class PolicyValueRNN(snt.RNNCore):
  """A recurrent multilayer perceptron with a value and a policy head."""

  def __init__(self, hidden_sizes: Sequence[int], num_actions: int):
    self._num_actions = num_actions
    super(PolicyValueRNN, self).__init__(name='policy_value_net')
    with self._enter_variable_scope():
      self._torso = snt.nets.MLP(hidden_sizes, activate_final=True, name='net')
      self._core = snt.LSTM(hidden_sizes[-1], name='rnn')
      self._policy_head = snt.Linear(num_actions, name='policy')
      self._value_head = snt.Linear(1, name='value')

  def _build(self, inputs_and_mask, state):
    inputs, mask = inputs_and_mask
    inputs = snt.BatchFlatten()(inputs)
    hiddens = self._torso(inputs)
    state = state._replace(hidden=state.hidden*mask, cell=state.cell*mask)
    lstm_output, next_state = self._core(inputs, state)
    lstm_output = tf.nn.relu(lstm_output) + hiddens  # skip connection
    logits = self._policy_head(lstm_output)
    value = self._value_head(lstm_output)
    return (logits, value), next_state

  @property
  def state_size(self):
    """Forward size(s) of state(s) used by the wrapped core."""
    return self._core.state_size

  @property
  def output_size(self):
    """Forward size of outputs produced by the wrapped core."""
    return (self._num_actions, 1)

  def initial_state(self, *args, **kwargs):
    """Creates the core initial state."""
    return self._core.initial_state(*args, **kwargs)


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray):
  """Initialize a DQN agent with default parameters."""
  network = PolicyValueRNN(
      hidden_sizes=[64, 64],
      num_actions=action_spec.num_values,
  )
  return ActorCriticRNN(
      obs_spec=obs_spec,
      action_spec=action_spec,
      network=network,
      optimizer=tf.train.AdamOptimizer(learning_rate=3e-3),
      sequence_length=32,
      td_lambda=0.9,
      agent_discount=0.99,
      seed=42,
  )
