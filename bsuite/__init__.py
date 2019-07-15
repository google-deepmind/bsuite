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
"""Set of experiments for investigating specific aspects of agent behaviour.

The behaviour suite (bsuite) is a set of targeted experiments designed to
investigate specific aspects (and scaling) of agent behaviour.

bsuite logs data "under the hood" to its own dataframe, and requires minimal
changes to your agent code to generate results.

Use case:

```python
import bsuite
env = bsuite.load_from_sweep('catch/0')
action = 0
env.step(action)
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import bsuite as _bsuite

load = _bsuite.load
load_from_id = _bsuite.load_from_id
