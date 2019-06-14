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
"""Reusable TestCase for environments in bsuite.

This generally kicks the tyres on an environment, and checks that it complies
with the interface contract for dm_env.Base.

Here we assume the environment simply exposes and accepts standalone arrays for
actions and observations, rather than handling the general case of arbitrarily
nested actions and observations.

When dm_env includes a general test case we can switch to use that.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl.testing import absltest
import dm_env


class EnvironmentTestMixin(object):
  """Mixin to help test implementations of dm_env.Base."""

  def setUp(self):
    super(EnvironmentTestMixin, self).setUp()
    self.environment = self.make_object_under_test()

  def make_object_under_test(self):
    raise NotImplementedError(
        "Attempt to run tests from an EnvironmentTestMixin subclass %s. "
        "Perhaps you forgot to override make_object_under_test?" % type(self))

  def tearDown(self):
    self.environment.close()
    # A call to super is required for cooperative multiple inheritance to work.
    super(EnvironmentTestMixin, self).tearDown()

  def test_reset(self):
    # Won't hurt to check this works twice in a row:
    for _ in range(2):
      self.reset_environment()

  def test_step_on_fresh_environment(self):
    # Action should be ignored here; ideally passing None should be acceptable
    # but environments don't always like this.
    step = self.step_environment()
    self.assertEqual(dm_env.StepType.FIRST, step.step_type,
                     "calling step() on a fresh environment must produce a "
                     "step with step_type FIRST")
    step = self.step_environment()
    self.assertNotEqual(
        dm_env.StepType.FIRST, step.step_type,
        "calling step() after a FIRST step must not produce another FIRST.")

  def test_step_after_reset(self):
    for _ in range(2):
      self.reset_environment()
      step = self.step_environment()
      self.assertNotEqual(
          dm_env.StepType.FIRST, step.step_type,
          "calling step() after a FIRST step must not produce another FIRST.")

  def test_longer_action_sequence(self):
    encountered_last_step = False
    for _ in range(2):
      self.reset_environment()
      prev_step_type = dm_env.StepType.FIRST
      for action in self.make_action_sequence():
        step = self.step_environment(action)
        if prev_step_type == dm_env.StepType.LAST:
          self.assertEqual(
              dm_env.StepType.FIRST, step.step_type,
              "step() must produce a FIRST step after a LAST step.")
        else:
          self.assertNotEqual(
              dm_env.StepType.FIRST, step.step_type,
              "step() must only produce a FIRST step after a LAST step "
              "or on a fresh environment.")
        if step.last():
          encountered_last_step = True
        prev_step_type = step.step_type
    if not encountered_last_step:
      logging.info(
          "Could not test the contract around end-of-episode behaviour. "
          "Consider implementing `make_action_sequence` so that an end of "
          "episode is reached.")
    else:
      logging.info("Successfully checked end of episode.")

  def make_action(self):
    spec = self.environment.action_spec()
    return spec.generate_value()

  def make_action_sequence(self):
    """Generate a sequence of actions for a longer test.

    Yields:
      A sequence of actions compatible with environment's action_spec().

    Ideally you should override this to generate an action sequence that will
    trigger an end of episode, in order to ensure this behaviour is tested.
    Otherwise it will just repeat a test value conforming to the action spec
    20 times.
    """
    for _ in range(20):
      yield self.make_action()

  def reset_environment(self):
    step = self.environment.reset()
    self.assertValidStep(step)
    self.assertEqual(dm_env.StepType.FIRST, step.step_type,
                     "reset() must produce a step with step_type FIRST")

  def step_environment(self, action=None):
    if action is None:
      action = self.make_action()
    step = self.environment.step(action)
    self.assertValidStep(step)
    return step

  def assertValidStep(self, step):
    self.assertIsInstance(step, dm_env.TimeStep)
    self.assertIsInstance(step.step_type, dm_env.StepType)

    if step.step_type is dm_env.StepType.FIRST:
      self.assertIsNone(step.reward, "a FIRST step must not have a reward.")
      self.assertIsNone(step.discount, "a FIRST step must not have a discount.")
    else:
      self.assertValidReward(step.reward)
      self.assertValidDiscount(step.discount)
    self.assertValidObservation(step.observation)

  def assertConformsToSpec(self, value, spec):
    try:
      spec.validate(value)
    except ValueError:
      self.fail("Invalid value: {}.".format(value))

  def assertValidObservation(self, observation):
    self.assertConformsToSpec(observation, self.environment.observation_spec())

  def assertValidReward(self, reward):
    self.assertConformsToSpec(reward, self.environment.reward_spec())

  def assertValidDiscount(self, discount):
    self.assertConformsToSpec(discount, self.environment.discount_spec())


if __name__ == "__main__":
  absltest.main()
