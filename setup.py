from setuptools import setup


setup(
    name='arm_numpyro_utilities',
    version="0.1.0",
    description="Helper functions for NumPyro",
    author="David Atlas",
    packages=['arm_numpyro_utilities'],
    install_requires=['numpyro', 'arviz', 'cufflinks', 'pandas']
)
