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
"""Example of generating a full set of bsuite results using multiprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

from absl import app
from absl import flags

from bsuite import bsuite
from bsuite import sweep
from bsuite.baselines import experiment
from bsuite.baselines.actor_critic import actor_critic
from bsuite.baselines.boot_dqn import boot_dqn
from bsuite.baselines.dqn import dqn
from bsuite.baselines.popart_dqn import popart_dqn
from bsuite.baselines.random import random
from bsuite.utils import sqlite_logging
from bsuite.utils import timer

flags.DEFINE_string('db_path', None, 'sqlite database path for results')
flags.DEFINE_integer('processes', None, 'number of processes')
flags.DEFINE_integer('num_episodes', 10000, 'number of episodes to run')
flags.DEFINE_enum('agent', 'random',
                  ['random', 'actor_critic', 'boot_dqn', 'dqn', 'popart_dqn'],
                  'which agent to run')

FLAGS = flags.FLAGS

_MAX_DB_INDEX = 100000

_AGENTS = {
    'actor_critic': actor_critic.default_agent,
    'boot_dqn': boot_dqn.default_agent,
    'dqn': dqn.default_agent,
    'popart_dqn': popart_dqn.default_agent,
    'random': random.default_agent,
}


@timer.time_run
def run(args):
  """Runs an agent against a single bsuite environment."""
  bsuite_id, num_episodes, db_path, agent_name = args
  print('Running {} and saving results to {}'.format(bsuite_id, db_path))
  experiment_name, setting_id = bsuite_id.split(sweep.SEPARATOR)
  raw_env = bsuite.load_from_id(bsuite_id)
  env = sqlite_logging.wrap_environment(
      raw_env, db_path, experiment_name, setting_id)

  agent = _AGENTS[agent_name](obs_spec=env.observation_spec(),
                              action_spec=env.action_spec())

  experiment.run(agent, env, num_episodes=num_episodes)


def _get_database_path():
  """Returns a path to a new sqlite database file, or FLAGS.db_path if set."""
  if FLAGS.db_path is not None:
    return FLAGS.db_path

  for i in range(_MAX_DB_INDEX):
    db_path = '/tmp/bsuite_demo_{}.db'.format(i)
    if not os.path.exists(db_path):
      break
  else:
    raise RuntimeError('Could not create a database file in /tmp')
  return db_path


def main(argv):
  del argv  # Unused.
  db_path = _get_database_path()
  args = [(s, FLAGS.num_episodes, db_path, FLAGS.agent) for s in sweep.SWEEP]
  pool = multiprocessing.Pool(FLAGS.processes)
  envname_and_duration = pool.map(run, args)
  df = timer.extract_df(envname_and_duration)
  csv_path = db_path.split('.')[0] + '_durations.csv'
  df.to_csv(csv_path)

if __name__ == '__main__':
  app.run(main)
