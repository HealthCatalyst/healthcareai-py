#!/usr/bin/env bash
################################
# ## Anaconda Build
#
# - Conda [Building Packages](https://conda.io/docs/building/build.html)
# - [Anaconda.org dashboard](https://anaconda.org/catalyst/healthcareai)
# - Taken from the excellent [conda.io docs](https://conda.io/docs/build_tutorials/pkgs.html)
# - Also, some taken from this [Travis CI build](https://gist.github.com/yoavram/05a3c04ddcf317a517d5)#
#
# ### Prereqs
#
# - Install conda build `conda install conda-build`
# - Install anaconda cli `conda install anaconda-client`
# - Login to anaconda.org with `anaconda login`
##################################

echo "You must be logged into anaconda by running anaconda login"

# configs
conda config --set always_yes true
conda config --set anaconda_upload no

# build skeleton from pypi package
conda skeleton pypi healthcareai

# Build for the main pythons
conda build --python 2.7 healthcareai
conda build --python 3.4 healthcareai
conda build --python 3.5 healthcareai
conda build --python 3.6 healthcareai

# Convert for all platforms (takes a while)
conda convert --platform all win-64/healthcareai-*-py*.tar.bz2 -o ~/repos/conda_test_1/builds/

# Upload to anaconda
# TODO parameterize the paths
anaconda upload ~/repos/conda_test_1/builds/**/healthcareai*.tar.bz2

# Clean up the mess
conda build purge

