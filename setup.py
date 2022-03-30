from setuptools import setup, find_packages
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='alt_gym_wordle',
    version=get_version('gym_wordle/_version.py'),
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
        'gym_wordle': ['*.txt']
    },
    install_requires=['gym', 'pandas', 'pygame']  # And any other dependencies foo needs
)