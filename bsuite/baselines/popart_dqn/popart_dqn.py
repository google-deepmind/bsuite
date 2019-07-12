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
"""A simple TensorFlow-based PopArt-DQN implementation.

References:
1. https://arxiv.org/pdf/1602.07714
2. https://arxiv.org/abs/1809.04474
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from bsuite.baselines import base
from bsuite.baselines import replay

import dm_env
from dm_env import specs
import numpy as np
import sonnet as snt
import tensorflow as tf
import trfl


class PopArtDQN(base.Agent):
  """A simple TensorFlow-based PopArt-DQN implementation."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.BoundedArray,
      torso: snt.AbstractModule,
      head: snt.Linear,
      target_torso: snt.AbstractModule,
      target_head: snt.Linear,
      batch_size: int,
      agent_discount: float,
      replay_capacity: int,
      min_replay_size: int,
      update_period: int,
      target_update_period: int,
      optimizer: tf.train.Optimizer,
      popart_step_size: float,
      popart_lb: float,
      popart_ub: float,
      epsilon: float,
      seed: int = None,
  ):
    """A simple DQN agent."""

    # PopArt-DQN configuration and hyperparameters.
    self._action_spec = action_spec
    self._num_actions = action_spec.maximum - action_spec.minimum + 1
    self._agent_discount = agent_discount
    self._batch_size = batch_size
    self._update_period = update_period
    self._target_update_period = target_update_period
    self._optimizer = optimizer
    self._popart_step_size = popart_step_size
    self._popart_lb = popart_lb
    self._popart_ub = popart_ub
    self._epsilon = epsilon
    self._total_steps = 0
    self._replay = replay.Replay(capacity=replay_capacity)
    self._min_replay_size = min_replay_size
    tf.set_random_seed(seed)
    self._rng = np.random.RandomState(seed)

    # popart statistics.
    self._mu = 0.  # first moment.
    self._nu = 1.  # second moment.
    self._target_mu = 0.  # target first moment.
    self._target_nu = 1.  # target second moment.

    # placeholders for injecting transitions into the graph.
    # `o` is the latest observation, used to select the next action;
    # `o_tm1`, `o_t` denote batches of consecutive observations.
    # `a_tm1` denotes the actions that were chosen in `o_tm1`
    # `r_t`, `d_t` denote the reward / discount associated with the transition.
    o = tf.placeholder(shape=(None,) + obs_spec.shape, dtype=obs_spec.dtype)
    o_tm1 = tf.placeholder(shape=(None,) + obs_spec.shape, dtype=obs_spec.dtype)
    a_tm1 = tf.placeholder(shape=(None,), dtype=action_spec.dtype)
    r_t = tf.placeholder(shape=(None,), dtype=tf.float32)
    d_t = tf.placeholder(shape=(None,), dtype=tf.float32)
    o_t = tf.placeholder(shape=(None,) + obs_spec.shape, dtype=obs_spec.dtype)
    # placeholders to inject the latest popart stats into the graph.
    mu = tf.placeholder(shape=[], dtype=tf.float32, name='mu')
    sigma = tf.placeholder(shape=[], dtype=tf.float32, name='sigma')
    target_mu = tf.placeholder(shape=[], dtype=tf.float32, name='tmu')
    target_sigma = tf.placeholder(shape=[], dtype=tf.float32, name='tsigma')

    network = snt.Sequential([torso, head])  # network
    q_normalized = network(o)  # normalized q-values.
    q = mu + sigma * q_normalized  # unnormalized q-values for behaviour.

    # normalized q-learning..
    target_network = snt.Sequential([target_torso, target_head])  # target net
    q_normalized_tm1, q_normalized_t = network(o_tm1), target_network(o_t)
    q_normalized_a_tm1 = trfl.batched_index(q_normalized_tm1, a_tm1)
    q_t = target_mu + target_sigma * q_normalized_t  # unnormalized target qs.
    q_t_max = tf.reduce_max(q_t, axis=1)  # now can max.
    target = r_t + agent_discount * d_t * q_t_max  # unnormalized to bootstrap.
    normalized_target = (target - target_mu) / target_sigma  # normalize target.
    loss = tf.square(  # compute loss in normalized space.
        tf.stop_gradient(normalized_target) - q_normalized_a_tm1)
    update_op = self._optimizer.minimize(loss)  # update.

    # target net updates.
    net_vars = network.get_all_variables()
    target_vars = target_network.get_all_variables()
    target_update_op = tf.group([
        target_var.assign(source_var)
        for target_var, source_var in zip(target_vars, net_vars)])

    # popart updates.
    target_q_normalized = target_network(o)
    target_q = target_mu + target_sigma * target_q_normalized
    bias, weights = head.get_variables()  # shape (A,) (H, A).
    bias_ph = tf.placeholder(shape=bias.shape, dtype=bias.dtype)
    weights_ph = tf.placeholder(shape=weights.shape, dtype=weights.dtype)
    assign_bw = tf.group(bias.assign(bias_ph), weights.assign(weights_ph))

    # make session and callables.
    session = tf.Session()
    self._value_fn = session.make_callable(
        q, [o, mu, sigma])  # values.
    self._target_value_fn = session.make_callable(
        target_q, [o, target_mu, target_sigma])  # target values.
    self._learn_fn = session.make_callable(  # learning.
        update_op, [
            mu, sigma, target_mu, target_sigma, o_tm1, a_tm1, r_t, d_t, o_t])
    self._target_update_fn = session.make_callable(target_update_op)
    self._eval_weights_fn = session.make_callable([bias, weights])  # get b,w.
    self._assign_weights_fn = session.make_callable(  # update b,w.
        assign_bw, [bias_ph, weights_ph])
    session.run(tf.global_variables_initializer())  # initialize all vars.

  def policy(self, timestep: dm_env.TimeStep) -> base.Action:
    """Select actions according to epsilon-greedy policy."""
    if self._rng.rand() < self._epsilon:  # sometimes, explore.
      return self._rng.randint(self._num_actions)

    sigma = self.sigma(self.mu, self.nu)  # compute stddev from 1st/2nd moments.
    q_values = self._value_fn(  # get unnormalized q-values.
        np.expand_dims(timestep.observation, axis=0), self.mu, sigma)
    return np.argmax(q_values[0])  # select greedy action, remove dummy batch.

  def update(self, old_step: dm_env.TimeStep, action: base.Action,
             new_step: dm_env.TimeStep):
    """Process a new transition from the environment and update model."""
    # store latest data in replay.
    self._replay.add([
        old_step.observation, action,
        new_step.reward, new_step.discount, new_step.observation])

    self._total_steps += 1
    if self._replay.size < self._min_replay_size:
      return  # if not enough data in replay, return.
    if self._total_steps % self._update_period != 0:
      return  # if not an update step, return.

    minibatch = self._replay.sample(self._batch_size)  # sample from replay.
    self._popart(*minibatch)  # apply popart.
    self._learn_fn(
        self.mu, self.sigma(self.mu, self.nu),
        self.target_mu, self.sigma(self.target_mu, self.target_nu),
        *minibatch)  # learn.

    if self._total_steps % self._target_update_period:
      self._target_update_fn()  # update target net vars
      self._target_mu = self.mu  # update target mean
      self._target_nu = self.nu  # update target shift

  def _popart(self, o_tm1, a_tm1, r_t, d_t, o_t):
    """Update statistics based on the latest batch of data."""
    # grab old stats.
    mu, nu, sigma = self.mu, self.nu, self.sigma(self.mu, self.nu)
    # compute new targets.
    q_next = self._target_value_fn(
        o_t, self.target_mu, self.sigma(self.target_mu, self.target_nu))
    target = r_t + self._agent_discount * d_t * np.max(q_next, axis=-1)
    # compute new stats.
    new_mu, new_nu = mu, nu
    for i in range(self._batch_size):
      new_mu += self._popart_step_size * (target[i] - new_mu)
      new_nu += self._popart_step_size * (target[i]**2 - new_nu)
    new_sigma = self.sigma(new_mu, new_nu)
    # preserve outputs under new stats.
    b, w = self._eval_weights_fn()
    b = (sigma * b + mu - new_mu) / new_sigma
    w = (sigma * w) / new_sigma
    self._assign_weights_fn(b, w)
    # store stats.
    self._mu = new_mu
    self._nu = new_nu

  def sigma(self, mu, nu):
    """Compute standard deviation from estimated first/second moments."""
    return np.sqrt(np.clip(nu - mu**2, self._popart_lb**2, self._popart_ub**2))

  @property
  def mu(self):
    return self._mu

  @property
  def nu(self):
    return self._nu

  @property
  def target_mu(self):
    return self._target_mu

  @property
  def target_nu(self):
    return self._target_nu


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray):
  """Initialize a PopArt-DQN agent with default parameters."""
  hidden_units = [20, 20]
  torso = snt.Sequential([
      snt.BatchFlatten(), snt.nets.MLP(hidden_units, activate_final=True)])
  head = snt.Linear(action_spec.num_values)
  target_torso = snt.Sequential([
      snt.BatchFlatten(), snt.nets.MLP(hidden_units, activate_final=True)])
  target_head = snt.Linear(action_spec.num_values)

  return PopArtDQN(
      obs_spec=obs_spec,
      action_spec=action_spec,
      torso=torso,
      head=head,
      target_torso=target_torso,
      target_head=target_head,
      batch_size=32,
      agent_discount=.99,
      replay_capacity=16384,
      min_replay_size=128,
      update_period=8,
      target_update_period=32,
      optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
      popart_step_size=1e-4,
      popart_lb=1e-5,
      popart_ub=1e5,
      epsilon=0.05,
      seed=42)
