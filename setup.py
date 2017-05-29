# -*- coding: utf-8 -*-
# from __future__ import unicode_literals
from setuptools import setup, find_packages


def readme():
    # I really prefer Markdown to reStructuredText.  PyPi does not.  This allows me
    # to have things how I'd like, but not throw complaints when people are trying
    # to install the package and they don't have pypandoc or the README in the
    # right place.
    # From https://coderwall.com/p/qawuyq/use-markdown-readme-s-in-python-modules
    try:
        import pypandoc
        long_description = pypandoc.convert('README.md', 'rst')
    except (IOError, ImportError):
        with open('README.md') as f:
            return f.read()
    else:
        return long_description

setup(name='healthcareai',
      version='1.0',
      maintainer='Levi Thatcher',
      maintainer_email='levi.thatcher@healthcatalyst.com',
      license='MIT',
      description='Tools for healthcare machine learning',
      keywords='machine learning healthcare data science',
      long_description=readme(),
      url='http://healthcare.ai',
      packages=find_packages(),
      install_requires=[
          'matplotlib>=1.5.3',
          'numpy>=1.11.2',
          'pandas>=0.19.0',
          # 'pyodbc>=3.0.10',
          'scipy>=0.18.1',
          'scikit-learn>=0.18',
          'imbalanced-learn>=0.2.1',
          'sqlalchemy>=1.1.5', 'sklearn'
      ],
      package_data={
          'examples': ['*.py', '*.ipynb']
      },
      tests_require=[
          'nose',
      ],
      test_suite='nose.collector',
      zip_safe=False,
      classifiers=[
          "Development Status :: 1 - Planning",
          "Intended Audience :: Healthcare Industry",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Operating System :: OS Independent",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.2",
          "Programming Language :: Python :: 3.3",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Information Analysis",
          "Topic :: Software Development :: Libraries :: Python Modules",
      ],
      include_package_data=True)
