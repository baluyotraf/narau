from setuptools import setup, find_packages

setup(
    name='narau',
    version='0.0.1',
    packages=find_packages(exclude=['tests.*', 'tests']),
    install_requires=[
        'numpy>=1.14.5',
    ],
    extras_require={
        'tf': ['tensorflow>=1.10.0'],
        'tf_gpu': ['tensorflow-gpu>=1.10.0'],
    },
)
