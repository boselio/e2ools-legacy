import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

extensions = [
    Extension(
        "e2ools.choice_fns",
        ["e2ools/choice_fns.pyx"],

    ),
]

setup(
    name = "e2ools",
    version = "0.0.1",
    author = "Brandon Oselio",
    author_email = "boselio@umich.edu",
    description = ("Edge Exchangeable Network Models and Inference"),
    license = "Apache 2.0",
    keywords = "edge exchangeability",
    packages = find_packages(),
    ext_modules = cythonize(extensions),
    long_description=read('README.md'),
)
