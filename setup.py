from distutils.core import setup

setup(name='HCPyTools',
      version='0.1.7',
      maintainer='Levi Thatcher',
      maintainer_email='levi.thatcher@healthcatalyst.com',
      description='Tools for healthcare data science',
      url='https://community.healthcatalyst.com/community/data-science',
      packages=['hcpytools', ],
      install_requires=[
         'numpy>=1.11.0',
         'scipy>=0.17.1',
         'scikit-learn>=0.17.1',
         'pandas>=0.18.1',
         'ceODBC>=2.0.1',
         'matplotlib>=1.5.0'
      ],
      scripts=['Example1', 'Example2'],
      classifiers="""license :: Copyright Health Catalyst 2016,
                     all rights reserved
                  """
      )
