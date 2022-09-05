from setuptools import setup, find_packages

setup(
    name='epop',
    version='0.1.0',
    url='https://github.com/vighnesh-nagpal/ePop.git',
    author='Vighnesh Nagpal',
    author_email='vighneshnagpal@berkeley.edu',
    description='Infer population-level eccentricity distributions using MCMC + hierarchical Bayesian modeling.',
    packages=find_packages(),    
    install_requires=['orbitize'],
)