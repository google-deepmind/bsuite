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
"""Run a Dqn agent instance (using TensorFlow 2) on a bsuite experiment."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from absl import app
from absl import flags

from bsuite import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.dqn_tf2 import dqn

import sonnet.v2 as snt
import tensorflow.compat.v2 as tf

flags.DEFINE_string('bsuite_id', 'catch/0', 'bsuite identifier')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'directory for csv logs')
flags.DEFINE_boolean('overwrite', False, 'overwrite csv if found')

flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 50, 'number of units per hidden layer')
flags.DEFINE_integer('batch_size', 32, 'size of batches sampled from replay')
flags.DEFINE_float('discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('replay_capacity', 100000, 'size of the replay buffer')
flags.DEFINE_integer('min_replay_size', 128, 'min replay size before training.')
flags.DEFINE_integer('sgd_period', 1, 'steps between online net updates')
flags.DEFINE_integer('target_update_period', 4,
                     'steps between target net updates')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate for optimizer')
flags.DEFINE_float('epsilon', 0.05, 'fraction of exploratory random actions')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.

  env = bsuite.load_and_record_to_csv(
      bsuite_id=FLAGS.bsuite_id,
      results_dir=FLAGS.results_dir,
      overwrite=FLAGS.overwrite,
  )

  # Making the networks.
  hidden_units = [FLAGS.num_units] * FLAGS.num_hidden_layers
  online_network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP(hidden_units + [env.action_spec().num_values]),
  ])
  target_network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP(hidden_units + [env.action_spec().num_values]),
  ])

  agent = dqn.DQNTF2(
      obs_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      online_network=online_network,
      target_network=target_network,
      batch_size=FLAGS.batch_size,
      discount=FLAGS.discount,
      replay_capacity=FLAGS.replay_capacity,
      min_replay_size=FLAGS.min_replay_size,
      sgd_period=FLAGS.sgd_period,
      target_update_period=FLAGS.target_update_period,
      optimizer=snt.optimizers.Adam(learning_rate=FLAGS.learning_rate),
      epsilon=FLAGS.epsilon,
      seed=FLAGS.seed,
  )

  experiment.run(
      agent=agent,
      environment=env,
      num_episodes=env.bsuite_num_episodes,  # pytype: disable=attribute-error
      verbose=FLAGS.verbose)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  app.run(main)
