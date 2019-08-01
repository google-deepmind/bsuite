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
from bsuite import sweep

from bsuite.baselines import experiment
from bsuite.baselines.dqn import dqn
from bsuite.baselines.utils import pool

import sonnet as snt
import tensorflow as tf

from typing import Text

# bsuite logging
flags.DEFINE_string('bsuite_id', 'catch/0',
                    'specify either a single bsuite_id (e.g. catch/0)\n'
                    'or a global variable from bsuite.sweep (e.g. SWEEP for '
                    'all of bsuite, or DEEP_SEA for just deep_sea experiment).')
flags.DEFINE_string('save_path', '/tmp/bsuite', 'where to save bsuite results')
flags.DEFINE_enum('logging_mode', 'csv', ['csv', 'sqlite'],
                  'which form of logging to use for bsuite results')
flags.DEFINE_boolean('overwrite', False, 'overwrite csv logging if found')

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


def run(bsuite_id: Text) -> Text:
  """Runs the agent against the environment specified by `bsuite_id`."""

  # Load the environment; here we opt for CSV logging.
  env = bsuite.load_and_record(
      bsuite_id=bsuite_id,
      save_path=FLAGS.save_path,
      logging_mode=FLAGS.logging_mode,
      overwrite=FLAGS.overwrite,
  )

  # Making the networks.
  hidden_units = [FLAGS.num_units] * FLAGS.num_hidden_layers
  online_network = snt.Sequential([
      snt.BatchFlatten(),
      snt.nets.MLP(hidden_units + [env.action_spec().num_values]),
  ])
  target_network = snt.Sequential([
      snt.BatchFlatten(),
      snt.nets.MLP(hidden_units + [env.action_spec().num_values]),
  ])

  agent = dqn.DQN(
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
      optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate),
      epsilon=FLAGS.epsilon,
      seed=FLAGS.seed,
  )

  experiment.run(
      agent=agent,
      environment=env,
      num_episodes=env.bsuite_num_episodes,  # pytype: disable=attribute-error
      verbose=FLAGS.verbose)

  return bsuite_id


def main(argv):
  """Parses whether to run a single bsuite_id, or multiprocess sweep."""
  del argv  # Unused.
  bsuite_id = FLAGS.bsuite_id

  if bsuite_id in sweep.SWEEP:
    print('Running a single bsuite_id={}'.format(bsuite_id))
    run(bsuite_id)

  elif hasattr(sweep, bsuite_id):
    bsuite_sweep = getattr(sweep, bsuite_id)
    print('Running a sweep over bsuite_id in sweep.{}'.format(bsuite_sweep))
    FLAGS.verbose = False
    pool.map_mpi(run, bsuite_sweep)

  else:
    raise ValueError('Invalid flag bsuite_id={}'.format(bsuite_id))


if __name__ == '__main__':
  app.run(main)
