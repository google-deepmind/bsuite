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
"""Run a Dqn agent instance on a bsuite experiment."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from absl import app
from absl import flags
from bsuite import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.popart_dqn import popart_dqn
import sonnet as snt

import tensorflow as tf

flags.DEFINE_string('bsuite_id', 'catch/0', 'bsuite identifier')
flags.DEFINE_integer('num_episodes', 10000, 'number of episodes to run')
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 256, 'number of units per hidden layer')
flags.DEFINE_integer('batch_size', 32, 'size of batches sampled from replay')
flags.DEFINE_float('agent_discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('replay_capacity', 10000, 'size of the replay buffer')
flags.DEFINE_integer('min_replay_size', 100, 'min replay size before training.')
flags.DEFINE_integer('update_period', 4, 'steps between online net updates')
flags.DEFINE_integer('target_update_period', 32, 'period of target net updates')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate for optimizer')
flags.DEFINE_float('popart_step_size', 1e-3, 'step size for stats updates')
flags.DEFINE_float('popart_lb', 1e-6, 'lower bound on standard deviation')
flags.DEFINE_float('popart_ub', 1e6, 'upper bound on standard deviation')
flags.DEFINE_float('epsilon', 0.05, 'fraction of exploratory random actions')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.
  env = bsuite.load_from_id(FLAGS.bsuite_id)

  hidden_units = [FLAGS.num_units] * FLAGS.num_hidden_layers
  torso = snt.Sequential(
      [snt.BatchFlatten(),
       snt.nets.MLP(hidden_units, activate_final=True)])
  head = snt.Linear(env.action_spec().num_values)
  target_torso = snt.Sequential(
      [snt.BatchFlatten(),
       snt.nets.MLP(hidden_units, activate_final=True)])
  target_head = snt.Linear(env.action_spec().num_values)

  agent = popart_dqn.PopArtDQN(
      obs_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      torso=torso,
      head=head,
      target_torso=target_torso,
      target_head=target_head,
      batch_size=FLAGS.batch_size,
      agent_discount=FLAGS.agent_discount,
      replay_capacity=FLAGS.replay_capacity,
      min_replay_size=FLAGS.min_replay_size,
      update_period=FLAGS.update_period,
      target_update_period=FLAGS.target_update_period,
      optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate),
      popart_step_size=FLAGS.popart_step_size,
      popart_lb=FLAGS.popart_lb,
      popart_ub=FLAGS.popart_ub,
      epsilon=FLAGS.epsilon,
      seed=FLAGS.seed)

  num_episodes = getattr(env, 'bsuite_num_episodes', FLAGS.num_episodes)

  experiment.run(
      agent=agent,
      environment=env,
      num_episodes=num_episodes,
      verbose=FLAGS.verbose)


if __name__ == '__main__':
  app.run(main)
