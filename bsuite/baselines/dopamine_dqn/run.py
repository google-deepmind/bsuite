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
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from absl import app
from absl import flags

from bsuite import bsuite
from bsuite.logging import terminal_logging
from bsuite.utils import gym_wrapper
from bsuite.utils import wrappers

from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import run_experiment

import tensorflow as tf

flags.DEFINE_string('bsuite_id', 'catch/0', 'bsuite identifier')
flags.DEFINE_integer('num_steps', 1000000, 'number of steps to run')
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
flags.DEFINE_string('base_dir', '/tmp/bsuite', 'directory for dopamine logs')
FLAGS = flags.FLAGS

OBSERVATION_SHAPE = (20, 20)


def main(_):

  def _create_network(num_actions, network_type, state):
    """Build deep network compatible with dopamine/discrete_domains/gym_lib."""
    x = tf.cast(state, tf.float32)
    x = tf.contrib.slim.flatten(x)
    for _ in range(FLAGS.num_hidden_layers):
      x = tf.contrib.slim.fully_connected(x, FLAGS.num_units)
    x = tf.contrib.slim.fully_connected(x, num_actions, activation_fn=None)
    return network_type(x)

  def create_agent(sess, environment, summary_writer=None):
    del summary_writer
    return dqn_agent.DQNAgent(
        sess=sess,
        num_actions=environment.action_space.n,
        observation_shape=OBSERVATION_SHAPE,
        observation_dtype=tf.float32,
        stack_size=1,
        network=_create_network,
        gamma=FLAGS.agent_discount,
        update_horizon=1,
        min_replay_history=FLAGS.min_replay_size,
        update_period=FLAGS.sgd_period,
        target_update_period=FLAGS.target_update_period,
        epsilon_decay_period=FLAGS.epsilon_decay_period,
        epsilon_train=FLAGS.epsilon,
        optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate),
    )

  def create_environment():
    env = bsuite.load_from_id(FLAGS.bsuite_id)
    env = wrappers.ImageObservation(env, OBSERVATION_SHAPE)
    if FLAGS.verbose:
      env = terminal_logging.wrap_environment(env, log_every=True)
    env = gym_wrapper.GymWrapper(env)
    env.game_over = False  # Dopamine looks for this
    return env

  runner = run_experiment.Runner(
      base_dir=FLAGS.base_dir,
      create_agent_fn=create_agent,
      create_environment_fn=create_environment,
      log_every_n=1,
      num_iterations=1,
      training_steps=1000000,  # Larger than strictly required for bsuite.
      evaluation_steps=0,
      max_steps_per_episode=1000,
  )
  runner.run_experiment()


if __name__ == '__main__':
  app.run(main)
