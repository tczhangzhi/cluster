from setuptools import setup, find_packages

requirements = [
    'numpy',
    'torch',
]

readme = open('README.rst').read()

setup(
    name='torchcluster',
    version='0.1.4',
    description='Torchcluster is a python package for cluster analysis.',
    license='MIT',
    author='Zhi Zhang',
    author_email='850734033@qq.com',
    keywords=['pytorch', 'cluster'],
    url='https://github.com/tczhangzhi/cluster',
    packages=find_packages(exclude=['tests']),
    long_description=readme,
    setup_requires=requirements
)