from setuptools import setup, find_packages

# Read the long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements from requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='webrock',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'rock=webrock.run:main',
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
)
