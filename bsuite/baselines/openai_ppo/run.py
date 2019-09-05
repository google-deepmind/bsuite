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
"""Trains an OpenAI Baselines PPO agent on bsuite.

Note that OpenAI Gym is not installed with bsuite by default.

See also github.com/openai/baselines for more information.
"""

# Import all packages

from absl import app
from absl import flags

from baselines.common.vec_env import dummy_vec_env
from baselines.ppo2 import ppo2

from bsuite import bsuite
from bsuite import sweep

from bsuite.baselines.utils import pool
from bsuite.logging import terminal_logging
from bsuite.utils import gym_wrapper

from typing import Text

# bsuite logging
flags.DEFINE_string('bsuite_id', 'catch/0',
                    'specify either a single bsuite_id (e.g. catch/0)\n'
                    'or a global variable from bsuite.sweep (e.g. SWEEP for '
                    'all of bsuite, or DEEP_SEA for just deep_sea experiment).')
flags.DEFINE_string('save_path', '/tmp/bsuite', 'where to save bsuite results')
flags.DEFINE_enum('logging_mode', 'csv', ['csv', 'sqlite', 'terminal'],
                  'which form of logging to use for bsuite results')
flags.DEFINE_boolean('overwrite', False, 'overwrite csv logging if found')

flags.DEFINE_boolean('verbose', True, 'whether to log to std output')
flags.DEFINE_string('network', 'mlp', 'name of network architecture')
flags.DEFINE_float('agent_discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('nsteps', 100, 'number of steps per ppo rollout')
flags.DEFINE_integer('total_timesteps', 1000000, 'total steps for experiment')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate for optimizer')

FLAGS = flags.FLAGS


def run(bsuite_id: Text) -> Text:
  """Runs a PPO agent on a given bsuite environment, logging to CSV."""

  def _load_env():
    raw_env = bsuite.load_and_record(
        bsuite_id=bsuite_id,
        save_path=FLAGS.save_path,
        logging_mode=FLAGS.logging_mode,
        overwrite=FLAGS.overwrite,
    )
    if FLAGS.verbose:
      raw_env = terminal_logging.wrap_environment(raw_env, log_every=True)
    return gym_wrapper.GymFromDMEnv(raw_env)
  env = dummy_vec_env.DummyVecEnv([_load_env])

  ppo2.learn(
      env=env,
      network=FLAGS.network,
      lr=FLAGS.learning_rate,
      total_timesteps=FLAGS.total_timesteps,  # make sure to run enough steps
      nsteps=FLAGS.nsteps,
      gamma=FLAGS.agent_discount,
  )

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
