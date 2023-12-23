import setuptools
import subprocess
import sys

REQUIREMENTS_FILE = 'requirements.txt'

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])

with open(REQUIREMENTS_FILE, 'r') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='src',
    install_requires=required,
    packages=setuptools.find_packages()
)