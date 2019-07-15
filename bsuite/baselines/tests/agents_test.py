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
"""Basic test coverage for agent training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from bsuite import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.actor_critic import actor_critic
from bsuite.baselines.actor_critic_rnn import actor_critic_rnn
from bsuite.baselines.boot_dqn import boot_dqn
from bsuite.baselines.dqn import dqn
from bsuite.baselines.popart_dqn import popart_dqn
from bsuite.baselines.random import random
import sonnet as snt
import tensorflow as tf


class AgentsTest(absltest.TestCase):

  def setUp(self):
    super(AgentsTest, self).setUp()
    self._env = bsuite.load_from_id('catch/0')
    self._action_spec = self._env.action_spec()
    self._obs_spec = self._env.observation_spec()
    self._num_actions = self._env.action_spec().num_values

  def test_actor_critic_feedforward(self):
    agent = actor_critic.default_agent(
        obs_spec=self._obs_spec,
        action_spec=self._action_spec,
    )
    experiment.run(agent, self._env, num_episodes=5)

  def test_actor_critic_recurrent(self):
    agent = actor_critic_rnn.default_agent(
        obs_spec=self._obs_spec,
        action_spec=self._action_spec,
    )
    experiment.run(agent, self._env, num_episodes=5)

  def test_boot_dqn(self):
    agent = boot_dqn.default_agent(
        obs_spec=self._obs_spec,
        action_spec=self._action_spec,
    )
    experiment.run(agent, self._env, num_episodes=5)

  def test_dqn(self):
    layer_sizes = [10, self._num_actions]
    net = snt.Sequential([snt.BatchFlatten(), snt.nets.MLP(layer_sizes)])
    agent = dqn.DQN(
        obs_spec=self._obs_spec, action_spec=self._action_spec,
        online_network=net, target_network=net,
        batch_size=5, agent_discount=.99, replay_capacity=20,
        min_replay_size=5, sgd_period=1, target_update_period=2,
        optimizer=tf.train.AdamOptimizer(0.01), epsilon=0.1, seed=42)
    experiment.run(agent, self._env, num_episodes=5)

  def test_popart_dqn(self):
    agent = popart_dqn.default_agent(
        obs_spec=self._obs_spec,
        action_spec=self._action_spec,
    )
    experiment.run(agent, self._env, num_episodes=5)

  def test_random(self):
    agent = random.Random(action_spec=self._env.action_spec(), seed=42)
    experiment.run(agent, self._env, num_episodes=5)


if __name__ == '__main__':
  absltest.main()
