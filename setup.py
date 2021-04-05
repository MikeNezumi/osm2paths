from setuptools import setup, find_packages

classifiers = [
  'Development Status :: 3 - Alpha',
  'Intended Audience :: Developers',
  'Operating System :: OS Independent',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]

setup(
  name='osm2paths',
  version='0.0.11',
  description='An automatic generator of JSON vector road network graph car tracks from OSM',
  long_description=open('README.md').read(),
  url='https://github.com/MikeNezumi/osm2tracks',
  author='MikeFreeman',
  author_email='michaelsvoboda42@gmail.com',
  license='MIT',
  classifiers=classifiers,
  keywords='json, traffic, simulation, data graphs, osm',
  py_modules=['osm2paths'],
  package_dir={'':'src'},
  packages=find_packages(where='src'),
  package_data={'src':['data/*.txt']},
  include_package_data=True,
  install_requires=[
    'utm==0.7.0',
    'pyglet==1.5.15',
    'pythematics==4.0.0'
  ]
)
