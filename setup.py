from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()
__version__ = '0.6.5'
setup(name='alt_gym_wordle',
    version=__version__,
    author='Ryan Snyder',
    description='OpenAI gym environment for training agents on Wordle',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ryan-snyder/gym-wordle',
    packages=find_packages(
        include=[
            'gym_wordle',
            'gym_wordle.*'
        ]
    ),
    package_data={
        'gym_wordle': ['envs/*.txt']
    },
    install_requires=['gym', 'pandas', 'pygame']  # And any other dependencies foo needs
)
