from setuptools import setup, find_packages

setup(
    name='marl_project',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pettingzoo[all]==1.24.0',
        'torch==2.0.1',
        'numpy==1.24.3',
        'matplotlib==3.7.1',
        'tensorboard==2.13.0',
        'pyyaml==6.0',
        'tqdm==4.65.0',
        'imageio==2.31.1',
        'pygame==2.5.0'
    ],
    entry_points={
        'console_scripts': [
            'marl-train=scripts.train:main',
            'marl-eval=scripts.evaluate:main',
            'marl-vis=scripts.visualize:main',
            'marl-demo=scripts.demo:main'
        ]
    }
)