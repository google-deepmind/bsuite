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

from importlib.machinery import SourceFileLoader

import setuptools

# Additional requirements for TensorFlow baselines, excluding OpenAI & Dopamine.
# See baselines/README.md for more information.
baselines_require = [
    'dm-sonnet',
    'dm-tree',
    'tensorflow',
    'tensorflow_probability',
    'trfl',
    'tqdm',
]

# Additional requirements for JAX baselines.
# See baselines/README.md for more information.
baselines_jax_require = [
    'dataclasses',
    'dm-haiku',
    'dm-tree',
    'jax',
    'jaxlib',
    'optax',
    'rlax',
    'tqdm',
]

baselines_third_party_require = [
    'tensorflow == 1.15',
    'dopamine-rl',
    'baselines',
]

testing_require = [
    'gym==0.20.0',
    'tensorflow_probability == 0.14.1',
    'mock',
    'pytest-xdist',
    'pytype',
]

setuptools.setup(
    name='bsuite',
    description=('Core RL Behaviour Suite. '
                 'A collection of reinforcement learning experiments.'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/deepmind/bsuite',
    author='DeepMind',
    author_email='dm-bsuite-eng+os@google.com',
    license='Apache License, Version 2.0',
    version=SourceFileLoader("_metadata", "bsuite/_metadata.py").load_module().__version__, 
    keywords='reinforcement-learning python machine-learning',
    packages=setuptools.find_packages(),
    install_requires=[
        'absl-py',
        'dm_env',
        'immutabledict',
        'matplotlib',
        'numpy',
        'pandas',
        'plotnine',
        'scipy',
        'scikit-image',
        'six',
        'termcolor',
    ],
    extras_require={
        'baselines': baselines_require,
        'baselines_jax': baselines_jax_require,
        'baselines_third_party': baselines_third_party_require,
        'testing': testing_require,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
