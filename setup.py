from setuptools import setup, find_packages

setup(name='tagi', 
    install_requires=['transformers>=4.1.1',
                      'torch>=1.7.0'],
    version='0.1.0',
    author="Harry Xie", 
    packages=find_packages(),
    include_package_data = True)
    