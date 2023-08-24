from setuptools import setup, find_packages

setup(
    name='autoagents',
    version='0.1.0',
    packages=find_packages(include=['agents', 'agents.*'])
)
