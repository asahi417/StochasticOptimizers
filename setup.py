from setuptools import setup, find_packages

FULL_VERSION = '0.0.0'

with open('README.md') as f:
    readme = f.read()

setup(
    name='stochastic_optimizers',
    version=FULL_VERSION,
    description='stochastic optimizer implementations for linear models',
    url='https://github.com/asahi417/StochasticOptimizers',
    long_description=readme,
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    packages=find_packages(exclude=('random', 'data')),
    include_package_data=True,
    test_suite='test',
    install_requires=[
        'sklearn',
        'scipy>=1.2.0,<2.0.0',
        'numpy',
        'cython',
        'matplotlib',
        'pandas',

    ]
)

