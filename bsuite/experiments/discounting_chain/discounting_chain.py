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
"""Simple diagnostic discounting challenge.

Observation is two pixels: (context, time_to_live)

Context will only be -1 in the first step, then equal to the action selected in
the first step. For all future decisions the agent is in a "chain" for that
action. Reward of +1 come  at one of: 1, 3, 10, 30, 100

However, depending on the seed, one of these chains has a 10% bonus.
"""

from bsuite.environments import discounting_chain

load = discounting_chain.DiscountingChain
