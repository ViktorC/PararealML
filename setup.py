from setuptools import setup, find_packages

setup(
    name='pararealml',
    version='0.3.0',
    description='A machine learning boosted parallel-in-time differential '
                'equation solver framework',
    url='https://github.com/ViktorC/PararealML',
    author='Viktor Csomor',
    author_email='viktor.csomor@gmail.com',
    license='MIT',
    install_requires=[
        'numpy>=1.21',
        'scipy>=1.7.0',
        'matplotlib>=3.5.0',
        'sympy>=1.6',
        'mpi4py>=3.0.0',
        'scikit-learn>=0.24.0',
        'tensorflow>=2.0.0',
        'tensorflow-probability>=0.10.0'
    ],
    packages=find_packages(exclude=('tests',)),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords=[
        'differential equations',
        'finite difference',
        'parallel-in-time',
        'parareal',
        'machine learning',
        'deep learning',
        'physics-informed neural networks',
        'scientific computing',
    ],
    include_package_data=True)
