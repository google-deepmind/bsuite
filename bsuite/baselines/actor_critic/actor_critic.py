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
"""A simple TensorFlow-based implementation of the actor-critic algorithm.

Reference: "Simple Statistical Gradient-Following Algorithms for Connectionist
            Reinforcement Learning" (Williams, 1992).

Link: http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf.
"""

from typing import Sequence, Tuple

from bsuite.baselines import base

import dm_env
import numpy as np
import sonnet as snt
import tensorflow.compat.v2 as tf
import trfl


class ActorCritic(base.Agent):
  """A simple TensorFlow-based feedforward actor-critic implementation."""

  def __init__(
      self,
      obs_spec: dm_env.specs.Array,
      network: snt.Module,
      optimizer: snt.Optimizer,
      sequence_length: int,
      td_lambda: float,
      discount: float,
      seed: int,
  ):
    """A simple actor-critic agent."""

    # Internalise hyperparameters.
    tf.random.set_seed(seed)
    self._sequence_length = sequence_length
    self._count = 0
    self._td_lambda = td_lambda
    self._discount = discount

    # Internalise network and optimizer.
    self._network = network
    policy_network = snt.Sequential([
        network,
        lambda pv: tf.random.categorical(pv[0], num_samples=1),
        lambda a: tf.cast(a, tf.int32),
        tf.squeeze,
    ])
    self._policy_network = tf.function(policy_network)
    self._optimizer = optimizer

    # Create windowed buffer for learning from trajectories.
    shapes = [obs_spec.shape, (), (), ()]
    dtypes = [obs_spec.dtype, np.int32, np.float32, np.float32]
    self._buffer = [
        np.zeros(shape=(self._sequence_length, 1) + shape, dtype=dtype)
        for shape, dtype in zip(shapes, dtypes)
    ]

  @tf.function
  def _step(self, transitions: Sequence[tf.Tensor]):
    """Do a batch of SGD on the actor + critic loss."""
    observations, actions, rewards, discounts, final_observation = transitions

    with tf.GradientTape() as tape:
      # Build actor and critic losses.
      logits, values = snt.BatchApply(self._network)(observations)
      _, bootstrap_value = self._network(final_observation)

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
    self._optimizer.apply(gradients, self._network.trainable_variables)

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    """Selects actions according to the latest softmax policy."""
    observation = tf.expand_dims(timestep.observation, axis=0)
    action = self._policy_network(observation)

    return action.numpy()

  def update(self, old_step: dm_env.TimeStep, action: base.Action,
             new_step: dm_env.TimeStep):
    """Receives a transition and performs a learning update."""

    # Insert this step into our rolling window 'batch'.
    items = [old_step.observation, action, new_step.reward, new_step.discount]
    for buf, item in zip(self._buffer, items):
      buf[self._count % self._sequence_length, 0] = item
    self._count += 1

    # When the batch is full, do a step of SGD.
    if self._count % self._sequence_length == 0:
      self._step(self._buffer + [new_step.observation[None, ...]])


class PolicyValueNet(snt.Module):
  """A simple multilayer perceptron with a value and a policy head."""

  def __init__(self, hidden_sizes: Sequence[int], num_actions: int):
    super().__init__(name='policy_value_net')
    self._torso = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP(hidden_sizes, activate_final=True),
    ])
    self._policy_head = snt.Linear(num_actions)
    self._value_head = snt.Linear(1)

  def __call__(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    embedding = self._torso(inputs)
    logits = self._policy_head(embedding)  # [B, A]
    value = tf.squeeze(self._value_head(embedding), axis=-1)  # [B]
    return logits, value


def default_agent(obs_spec: dm_env.specs.Array,
                  action_spec: dm_env.specs.DiscreteArray):
  """Initialize a DQN agent with default parameters."""
  network = PolicyValueNet(
      hidden_sizes=[64, 64],
      num_actions=action_spec.num_values,
  )
  return ActorCritic(
      obs_spec=obs_spec,
      network=network,
      optimizer=snt.optimizers.Adam(learning_rate=1e-2),
      sequence_length=32,
      td_lambda=0.9,
      discount=0.99,
      seed=42,
  )
