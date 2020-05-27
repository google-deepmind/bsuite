#!/bin/bash

# Fail on any error.
set -e
# Display commands being run.
set -x

cd ~/
rm -rf bsuite/

# Similar to installation instructions in README.
# 1. Clone repo.
git clone https://github.com/deepmind/bsuite

# 2. Set up virtual environment.
python3 -m venv bsuite_testing
source bsuite_testing/bin/activate

# 3. Pip install.
pip install --upgrade pip setuptools
pip install bsuite/

# Check environment tests & that we can import and instantiate envs.
pip install nose
nosetests bsuite/bsuite/tests/environments_test.py
python3 -c "import bsuite
env = bsuite.load_from_id('catch/0')
env.reset()"

# Check that we can install + import baselines.
pip install bsuite[baselines_jax]
python3 -c "from bsuite.baselines.jax import dqn"

pip install bsuite[baselines]
python3 -c "from bsuite.baselines.tf import dqn"

deactivate
rm -rf bsuite_testing/
