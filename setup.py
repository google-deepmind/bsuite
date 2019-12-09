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
"""Install script for setuptools."""

# Import all required packages

from setuptools import find_packages
from setuptools import setup

# Additional requirements for bsuite/baselines, excluding OpenAI, Dopamine and
# JAX examples.
baselines_require = [
    'dm-sonnet',
    'dm-tree',
    # trfl needs to be in sync with a given TensorFlow version. We depend on
    # TensorFlow transitively via the "tensorflow" extras_require entry
    # specified by trfl.
    'trfl[tensorflow]',
    'tqdm',
]

dependency_links = [
    'http://github.com/google/dopamine/tarball/master#egg=dopamine_rl',
]


setup(
    name='bsuite',
    description=('Core RL Behaviour Suite. '
                 'A collection of reinforcement learning experiments.'),
    author='DeepMind',
    license='Apache License, Version 2.0',
    keywords='reinforcement-learning python machine learning',
    packages=find_packages(),
    install_requires=[
        'absl-py',
        'dm_env',
        'matplotlib',
        'numpy',
        'pandas',
        'plotnine',
        'scipy',
        'scikit-image',
        'six',
        'termcolor',
    ],
    tests_require=[
        'absl-py',
        'nose',
    ],
    extras_require={
        'baselines': baselines_require,
    },
    test_suite='nose.collector',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
