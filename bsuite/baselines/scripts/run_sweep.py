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
"""Example of generating a full set of bsuite results using multiprocessing."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

import multiprocessing
import os

from absl import app
from absl import flags

from bsuite import bsuite
from bsuite import sweep
from bsuite.baselines import experiment
from bsuite.baselines.actor_critic import actor_critic
from bsuite.baselines.actor_critic_rnn import actor_critic_rnn
from bsuite.baselines.boot_dqn import boot_dqn
from bsuite.baselines.dqn import dqn
from bsuite.baselines.popart_dqn import popart_dqn
from bsuite.baselines.random import random

import termcolor
import tqdm
from typing import Text, Tuple

flags.DEFINE_string('db_path', None, 'sqlite database path for results')
flags.DEFINE_integer('processes', None, 'number of processes')
flags.DEFINE_enum('agent', 'random', [
    'actor_critic',
    'actor_critic_rnn',
    'boot_dqn',
    'dqn',
    'popart_dqn',
    'random',
], 'which agent to run')

FLAGS = flags.FLAGS

_AGENTS = {
    'actor_critic': actor_critic.default_agent,
    'actor_critic_rnn': actor_critic_rnn.default_agent,
    'boot_dqn': boot_dqn.default_agent,
    'dqn': dqn.default_agent,
    'popart_dqn': popart_dqn.default_agent,
    'random': random.default_agent,
}


def run(run_info: Tuple[Text, Text, Text]) -> Text:
  """Runs an agent against a single bsuite environment."""
  bsuite_id, db_path, agent_name = run_info

  # Create the environment and retrieve its spec.
  env = bsuite.load_and_record_to_sqlite(bsuite_id, db_path)
  obs_spec = env.observation_spec()
  action_spec = env.action_spec()

  # Create the agent and run the experiment.
  agent = _AGENTS[agent_name](obs_spec=obs_spec, action_spec=action_spec)
  experiment.run(
      agent=agent,
      environment=env,
      num_episodes=env.bsuite_num_episodes,
  )  # pytype: disable=attribute-error

  return bsuite_id


def _get_database_path(max_db_index: int = 10000) -> Text:
  """Returns a path to a new sqlite database file."""
  for i in range(max_db_index):
    path = '/tmp/bsuite_demo_{}.db'.format(i)
    if not os.path.exists(path):
      return path
  raise RuntimeError('Could not create a database file in /tmp.')


def main(argv):
  del argv  # Unused.
  db_path = FLAGS.db_path or _get_database_path()
  agent_name = FLAGS.agent

  # Note that `sweep.SWEEP` contains the ids for all of the bsuite experiments.
  run_args = [(bsuite_id, db_path, agent_name) for bsuite_id in sweep.SWEEP]

  num_experiments = len(run_args)
  num_processes = FLAGS.processes or multiprocessing.cpu_count()

  message = """
  Experiment info
  ---------------
  Agent: {agent_name}
  Num experiments: {num_experiments}
  Num worker processes: {num_processes}
  """.format(
      agent_name=agent_name,
      num_processes=num_processes,
      num_experiments=num_experiments)
  termcolor.cprint(message, color='blue', attrs=['bold'])

  # Create a pool of processes, dispatch the experiments to them, show progress.
  pool = multiprocessing.Pool(num_processes)
  progress_bar = tqdm.tqdm(total=num_experiments)
  for bsuite_id in pool.imap(run, run_args):
    description = '[Last finished: {}]'.format(bsuite_id)
    progress_bar.set_description(termcolor.colored(description, color='green'))
    progress_bar.update()


if __name__ == '__main__':
  app.run(main)
