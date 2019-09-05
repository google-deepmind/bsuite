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
"""Trains an Dopamine DQN agent on bsuite.

Note that Dopamine is not installed with bsuite by default.

See also github.com/google/dopamine for more information.
"""

# Import all packages

from absl import app
from absl import flags

from bsuite import bsuite
from bsuite import sweep

from bsuite.baselines.utils import pool
from bsuite.logging import terminal_logging
from bsuite.utils import gym_wrapper
from bsuite.utils import wrappers

from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import run_experiment

import gym
import tensorflow as tf

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

# algorithm
flags.DEFINE_integer('training_steps', 1000000, 'number of steps to run')
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 50, 'number of units per hidden layer')
flags.DEFINE_float('agent_discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('replay_capacity', 100000, 'size of the replay buffer')
flags.DEFINE_integer('min_replay_size', 128, 'min replay size before training.')
flags.DEFINE_integer('sgd_period', 1, 'steps between online net updates')
flags.DEFINE_integer('target_update_period', 4,
                     'steps between target net updates')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate for optimizer')
flags.DEFINE_float('epsilon', 0.05, 'fraction of exploratory random actions')
flags.DEFINE_float('epsilon_decay_period', 1000,
                   'number of steps to anneal epsilon')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')
flags.DEFINE_string('base_dir', '/tmp/dopamine', 'directory for dopamine logs')
FLAGS = flags.FLAGS

OBSERVATION_SHAPE = (20, 20)


def run(bsuite_id: Text) -> Text:
  """Runs Dopamine DQN on a given bsuite environment, logging to CSV."""

  class Network(tf.keras.Model):
    """Build deep network compatible with dopamine/discrete_domains/gym_lib."""

    def __init__(self, num_actions: int, name='Network'):
      super(Network, self).__init__(name=name)
      self.forward_fn = tf.keras.Sequential(
          [tf.keras.layers.Flatten()] +
          [tf.keras.layers.Dense(FLAGS.num_units,
                                 activation=tf.keras.activations.relu)
           for _ in range(FLAGS.num_hidden_layers)] +
          [tf.keras.layers.Dense(num_actions, activation=None)])

    def call(self, state):
      """Creates the output tensor/op given the state tensor as input."""
      x = tf.cast(state, tf.float32)
      x = self.forward_fn(x)
      return atari_lib.DQNNetworkType(x)

  def create_agent(sess: tf.Session, environment: gym.Env, summary_writer=None):
    """Factory method for agent initialization in Dopmamine."""
    del summary_writer
    return dqn_agent.DQNAgent(
        sess=sess,
        num_actions=environment.action_space.n,
        observation_shape=OBSERVATION_SHAPE,
        observation_dtype=tf.float32,
        stack_size=1,
        network=Network,
        gamma=FLAGS.agent_discount,
        update_horizon=1,
        min_replay_history=FLAGS.min_replay_size,
        update_period=FLAGS.sgd_period,
        target_update_period=FLAGS.target_update_period,
        epsilon_decay_period=FLAGS.epsilon_decay_period,
        epsilon_train=FLAGS.epsilon,
        optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate),
    )

  def create_environment() -> gym.Env:
    """Factory method for environment initialization in Dopmamine."""
    env = bsuite.load_and_record(
        bsuite_id=bsuite_id,
        save_path=FLAGS.save_path,
        logging_mode=FLAGS.logging_mode,
        overwrite=FLAGS.overwrite,
    )
    env = wrappers.ImageObservation(env, OBSERVATION_SHAPE)
    if FLAGS.verbose:
      env = terminal_logging.wrap_environment(env, log_every=True)
    env = gym_wrapper.GymFromDMEnv(env)
    env.game_over = False  # Dopamine looks for this
    return env

  runner = run_experiment.Runner(
      base_dir=FLAGS.base_dir,
      create_agent_fn=create_agent,
      create_environment_fn=create_environment,
      log_every_n=1,
      num_iterations=1,
      training_steps=FLAGS.training_steps,  # Make sure large enough for bsuite
      evaluation_steps=0,
      max_steps_per_episode=1000,
  )
  runner.run_experiment()

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
