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
"""A simple TensorFlow-based implementation of the actor-critic algorithm.

Reference: "Simple Statistical Gradient-Following Algorithms for Connectionist
            Reinforcement Learning" (Williams, 1992).

Link: http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.baselines import base
import dm_env
from dm_env import specs
import numpy as np
import sonnet as snt
import tensorflow as tf
import trfl
from typing import Sequence, Tuple

nest = tf.contrib.framework.nest


class ActorCritic(base.Agent):
  """A simple TensorFlow-based feedforward actor-critic implementation."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      network: snt.AbstractModule,
      optimizer: tf.train.Optimizer,
      sequence_length: int,
      td_lambda: int,
      agent_discount: float,
      seed: int,
  ):
    """A simple actor-critic agent."""
    tf.set_random_seed(seed)
    self._sequence_length = sequence_length
    self._count = 0

    # Create the policy ops..
    obs = tf.placeholder(shape=obs_spec.shape, dtype=obs_spec.dtype)
    online_logits, _ = network(tf.expand_dims(obs, 0))
    action = tf.squeeze(tf.multinomial(online_logits, 1, output_dtype=tf.int32))

    # Create placeholders and numpy arrays for learning from trajectories.
    shapes = [obs_spec.shape, (), (), ()]
    dtypes = [obs_spec.dtype, np.int32, np.float32, np.float32]

    placeholders = [
        tf.placeholder(shape=(self._sequence_length, 1) + shape, dtype=dtype)
        for shape, dtype in zip(shapes, dtypes)]
    observations, actions, rewards, discounts = placeholders

    self.arrays = [
        np.zeros(shape=(self._sequence_length, 1) + shape, dtype=dtype)
        for shape, dtype in zip(shapes, dtypes)]

    # Build actor and critic losses.
    logits, values = snt.BatchApply(network)(observations)
    _, bootstrap_value = network(tf.expand_dims(obs, 0))

    critic_loss, (advantages, _) = trfl.td_lambda(
        state_values=values,
        rewards=rewards,
        pcontinues=agent_discount * discounts,
        bootstrap_value=bootstrap_value,
        lambda_=td_lambda)
    actor_loss = trfl.discrete_policy_gradient_loss(logits, actions, advantages)
    train_op = optimizer.minimize(actor_loss + critic_loss)

    # Create TF session and callables.
    session = tf.Session()
    self._policy_fn = session.make_callable(action, [obs])
    self._update_fn = session.make_callable(train_op, placeholders + [obs])
    session.run(tf.global_variables_initializer())

  def policy(self, timestep: dm_env.TimeStep) -> base.Action:
    """Selects actions according to the latest softmax policy."""
    return np.int32(self._policy_fn(timestep.observation))

  def update(self, old_step: dm_env.TimeStep, action: base.Action,
             new_step: dm_env.TimeStep):
    """Receives a transition and performs a learning update."""

    # Insert this step into our rolling window 'batch'.
    items = [old_step.observation, action, new_step.reward, new_step.discount]
    for buf, item in zip(self.arrays, items):
      buf[self._count % self._sequence_length, 0] = item
    self._count += 1

    # When the batch is full, do a step of SGD.
    if self._count % self._sequence_length == 0:
      self._update_fn(*(self.arrays + [new_step.observation]))


class PolicyValueNet(snt.AbstractModule):
  """A simple multilayer perceptron with a value and a policy head."""

  def __init__(self, hidden_sizes: Sequence[int], num_actions: int):
    self._num_actions = num_actions
    self._hidden_sizes = hidden_sizes
    super(PolicyValueNet, self).__init__(name='policy_value_net')

  def _build(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    inputs = snt.BatchFlatten()(inputs)
    hiddens = snt.nets.MLP(self._hidden_sizes, activate_final=True)(inputs)
    logits = snt.Linear(self._num_actions)(hiddens)
    value = tf.squeeze(snt.Linear(1)(hiddens), axis=-1)
    return logits, value


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray,
                  num_hidden_layers: int = 2,
                  num_units: int = 256,
                  **kwargs):
  """Initialize a DQN agent with default parameters."""

  params = {
      'agent_discount': .99,
      'sequence_length': 32,
      'td_lambda': 0.9,
      'optimizer': tf.train.AdamOptimizer(learning_rate=1e-2),
      'seed': 42,
  }
  params.update(kwargs)

  num_actions = action_spec.num_values
  units_per_hidden_layer = [num_units] * num_hidden_layers

  network = PolicyValueNet(units_per_hidden_layer, num_actions)

  return ActorCritic(
      obs_spec=obs_spec,
      action_spec=action_spec,
      network=network,
      **params)
