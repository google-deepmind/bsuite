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
from bsuite.baselines.utils import sequence

import dm_env
from dm_env import specs
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import trfl

tfd = tfp.distributions


class ActorCritic(base.Agent):
  """A simple TensorFlow-based feedforward actor-critic implementation."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.Array,
      network: 'PolicyValueNet',
      optimizer: snt.Optimizer,
      max_sequence_length: int,
      td_lambda: float,
      discount: float,
      seed: int,
  ):
    """A simple actor-critic agent."""

    # Internalise hyperparameters.
    tf.random.set_seed(seed)
    self._td_lambda = td_lambda
    self._discount = discount

    # Internalise network and optimizer.
    self._network = network
    self._optimizer = optimizer

    # Create windowed buffer for learning from trajectories.
    self._buffer = sequence.Buffer(obs_spec, action_spec, max_sequence_length)

  @tf.function
  def _sample_policy(self, inputs: tf.Tensor) -> tf.Tensor:
    policy, _ = self._network(inputs)
    action = policy.sample()
    return tf.squeeze(action)

  @tf.function
  def _step(self, trajectory: sequence.Trajectory):
    """Do a batch of SGD on the actor + critic loss."""
    observations, actions, rewards, discounts = trajectory

    # Add dummy batch dimensions.
    rewards = tf.expand_dims(rewards, axis=-1)  # [T, 1]
    discounts = tf.expand_dims(discounts, axis=-1)  # [T, 1]
    observations = tf.expand_dims(observations, axis=1)  # [T+1, 1, ...]

    # Extract final observation for bootstrapping.
    observations, final_observation = observations[:-1], observations[-1]

    with tf.GradientTape() as tape:
      # Build actor and critic losses.
      policies, values = snt.BatchApply(self._network)(observations)
      _, bootstrap_value = self._network(final_observation)

      critic_loss, (advantages, _) = trfl.td_lambda(
          state_values=values,
          rewards=rewards,
          pcontinues=self._discount * discounts,
          bootstrap_value=bootstrap_value,
          lambda_=self._td_lambda)
      advantages = tf.squeeze(advantages, axis=-1)  # [T]
      actor_loss = -policies.log_prob(actions) * tf.stop_gradient(advantages)
      loss = tf.reduce_sum(actor_loss) + critic_loss

    gradients = tape.gradient(loss, self._network.trainable_variables)
    self._optimizer.apply(gradients, self._network.trainable_variables)

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    """Selects actions according to the latest softmax policy."""
    observation = tf.expand_dims(timestep.observation, axis=0)
    action = self._sample_policy(observation)

    return action.numpy()

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
  ):
    """Receives a transition and performs a learning update."""

    self._buffer.append(timestep, action, new_timestep)

    # When the batch is full, do a step of SGD.
    if self._buffer.full() or new_timestep.last():
      trajectory = self._buffer.drain()
      trajectory = tree.map_structure(tf.convert_to_tensor, trajectory)
      self._step(trajectory)


class PolicyValueNet(snt.Module):
  """A simple multilayer perceptron with a value and a policy head."""

  def __init__(self, hidden_sizes: Sequence[int],
               action_spec: specs.DiscreteArray):
    super().__init__(name='policy_value_net')
    self._torso = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP(hidden_sizes, activate_final=True),
    ])
    self._policy_head = snt.Linear(action_spec.num_values)
    self._value_head = snt.Linear(1)
    self._action_dtype = action_spec.dtype

  def __call__(self, inputs: tf.Tensor) -> Tuple[tfd.Distribution, tf.Tensor]:
    """Returns a (policy, value) pair: (pi(.|s), V(s))."""
    embedding = self._torso(inputs)
    logits = self._policy_head(embedding)  # [B, A]
    value = tf.squeeze(self._value_head(embedding), axis=-1)  # [B]
    policy = tfd.Categorical(logits, dtype=self._action_dtype)
    return policy, value


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray) -> base.Agent:
  """Initialize a DQN agent with default parameters."""
  network = PolicyValueNet(
      hidden_sizes=[64, 64],
      action_spec=action_spec,
  )
  return ActorCritic(
      obs_spec=obs_spec,
      action_spec=action_spec,
      network=network,
      optimizer=snt.optimizers.Adam(learning_rate=3e-3),
      max_sequence_length=32,
      td_lambda=0.9,
      discount=0.99,
      seed=42,
  )
