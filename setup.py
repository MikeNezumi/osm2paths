from setuptools import setup, find_packages

classifiers = [
  'Development Status :: 3 - Alpha',
  'Intended Audience :: Developers',
  'Operating System :: OS Independent',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]

setup(
  name='osm2tracks',
  version='0.0.1',
  description='An automtic generator of JSON vector road network graph car tracks from OSM',
  long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='https://github.com/MikeNezumi/osm2tracks',
  author='MikeFreeman',
  author_email='michaelsvoboda42@gmail.com',
  license='MIT',
  classifiers=classifiers,
  keywords='json, traffic, simulation, data graphs, osm',
  package_dir={'':'src'},
  py_modules=["osm2tracks"],
  install_requires=['utm', 'json', 'pyglet', 'subprocess', 'pythematics']
)
