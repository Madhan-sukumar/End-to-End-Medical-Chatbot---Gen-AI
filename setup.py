from setuptools import find_packages, setup

setup (
    name = 'Generative AI Project',
    version= '0.0.0',
    author = 'Madhan',
    packages = find_packages(), #look for __init__.py, whenever it is present, it consider the folder as local package
    install_requires = []
)