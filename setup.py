# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from distutils.core import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='HCPyTools',
      version='0.1.7',
      maintainer='Levi Thatcher',
      maintainer_email='levi.thatcher@healthcatalyst.com',
      license='proprietary',
      description='Tools for healthcare data science',
      long_description=readme(),
      url='https://community.healthcatalyst.com/community/data-science',
      packages=[
          'hcpytools',
      ],
      install_requires=[
          'numpy>=1.11.0',
          'scipy>=0.17.1',
          'scikit-learn>=0.17.1',
          'pandas>=0.18.1',
          'ceODBC>=2.0.1',
          'matplotlib>=1.5.0'
      ],
      tests_require=[
          'nose',
      ],
      test_suite='nose.collector',
      zip_safe=False,
      classifiers=[
          "Development Status :: 1 - Planning",
          "Intended Audience :: Healthcare Industry",
          "Intended Audience :: Developers",
          "License :: Other/Proprietary License",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.2",
          "Programming Language :: Python :: 3.3",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Software Development :: Libraries :: Python Modules",
      ],)
