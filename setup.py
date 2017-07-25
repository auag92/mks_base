#!/usr/bin/env python

import subprocess
from setuptools import setup, find_packages
import os


setup(name='mks_base',
      version=0.1,
      description='Materials Knowledge Systems (MKS) code repository',
      author='Apaar Shanker',
      author_email='apaar92@gmail.com',
      url='',
      packages=find_packages(),
      package_data={'': ['tests/*.py']},
      install_requires=['scipy', 'numpy', 'scikit-learn', 'toolz', 'matplotlib'],
      data_files=['setup.cfg'])

