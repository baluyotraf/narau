from setuptools import setup, find_packages

setup(
    name='iv-model',
    version='0.0.1',
    packages=find_packages(exclude=['tests.*', 'tests']),
    install_requires=[
    ],
)
