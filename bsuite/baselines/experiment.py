# python3
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
"""A simple agent-environment training loop."""

from bsuite.baselines import base
from bsuite.logging import terminal_logging

import dm_env


def run(agent: base.Agent,
        environment: dm_env.Environment,
        num_episodes: int,
        verbose: bool = False) -> None:
  """Runs an agent on an environment.

  Note that for bsuite environments, logging is handled internally.

  Args:
    agent: The agent to train and evaluate.
    environment: The environment to train on.
    num_episodes: Number of episodes to train for.
    verbose: Whether to also log to terminal.
  """

  if verbose:
    environment = terminal_logging.wrap_environment(
        environment, log_every=True)  # pytype: disable=wrong-arg-types

  for _ in range(num_episodes):
    # Run an episode.
    timestep = environment.reset()
    while not timestep.last():
      # Generate an action from the agent's policy.
      action = agent.select_action(timestep)

      # Step the environment.
      new_timestep = environment.step(action)

      # Tell the agent about what just happened.
      agent.update(timestep, action, new_timestep)

      # Book-keeping.
      timestep = new_timestep
