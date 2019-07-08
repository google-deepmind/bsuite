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
"""Run agent on a bsuite experiment."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from absl import app
from absl import flags
from bsuite import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.boot_dqn import boot_dqn
import tensorflow as tf

# Network options
flags.DEFINE_integer('num_ensemble', 16, 'number of ensemble networks')
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 256, 'number of units per hidden layer')
flags.DEFINE_float('prior_scale', 1., 'scale for additive prior network')

# Core DQN options
flags.DEFINE_integer('batch_size', 32, 'size of batches sampled from replay')
flags.DEFINE_float('agent_discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('replay_capacity', 16384, 'size of the replay buffer')
flags.DEFINE_integer('min_replay_size', 128, 'min transitions for sampling')
flags.DEFINE_integer('sgd_period', 16, 'steps between online net updates')
flags.DEFINE_float('mask_prob', 0.5, 'probability for bootstrap mask')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate for optimizer')
flags.DEFINE_float('epsilon', 0.05, 'fraction of exploratory random actions')

# Experiment options
flags.DEFINE_string('bsuite_id', 'catch/0', 'bsuite identifier')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')
flags.DEFINE_integer('num_episodes', 10000, 'number of episodes to run')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.
  env = bsuite.load_from_id(FLAGS.bsuite_id)
  ensemble = boot_dqn.make_ensemble(
      num_ensemble=FLAGS.num_ensemble,
      num_hidden_layers=FLAGS.num_hidden_layers,
      num_units=FLAGS.num_units,
      num_actions=env.action_spec().num_values,
      prior_scale=FLAGS.prior_scale)

  agent = boot_dqn.BootstrappedDqn(
      obs_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      ensemble=ensemble,
      batch_size=FLAGS.batch_size,
      agent_discount=FLAGS.agent_discount,
      replay_capacity=FLAGS.replay_capacity,
      min_replay_size=FLAGS.min_replay_size,
      sgd_period=FLAGS.sgd_period,
      optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate),
      epsilon_fn=lambda x: FLAGS.epsilon,
      seed=FLAGS.seed)

  FLAGS.alsologtostderr = True
  experiment.run(
      agent, env, num_episodes=FLAGS.num_episodes, verbose=FLAGS.verbose)


if __name__ == '__main__':
  app.run(main)
