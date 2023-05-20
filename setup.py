from setuptools import setup, find_packages


setup(
    name='hanz',
    version='0.0.1',
    license='Apache 2.0',
    author="Yuhan Zhang",
    author_email='yuhanz@gmail.com',
    packages=find_packages(),
    package_dir={'': './'},
    url='https://github.com/yuhanz/hanz',
    keywords='Neural Network DSL',
    install_requires=[
          'torch',
          'torchvision'
      ],

)
