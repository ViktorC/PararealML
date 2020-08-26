from setuptools import setup
from setuptools import find_packages

setup(
    name='PararealML',
    version='0.0.1',
    description='A parallel-in-time differential equation solver framework '
                'with machine learning acceleration',
    url='https://github.com/ViktorC/PararealML',
    author='Viktor Csomor',
    author_email='viktor.csomor@gmail.com',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False)
