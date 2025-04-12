'''
Description:  
Author: Huang J
Date: 2025-04-10 09:29:26
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

NAME = "chunk-factory"
DESCRIPTION = "An efficient chunking library that integrates traditional and advanced methods, with real-time evaluation of chunking results."
URL = "https://github.com/hjandlm/Chunk-Factory"
EMAIL = "hjie97bi@gmail.com"
AUTHOR = "Jie Huang"
REQUIRES_PYTHON = ">=3.10.0"
VERSION = "0.1.0"

REQUIRED = ['transformers','sentence-transformers','tiktoken','pandas','openai','google-genai','seaborn','matplotlib','jieba','nltk']

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION
    
with io.open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:    
    requirements = f.read().splitlines()


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Pushing git tags…")
        os.system("git push --tags")

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        sys.exit()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    install_requires=requirements,
    url=URL,
    packages=find_packages(),
    keywords = ['python','windows','mac','linux','text chunk','chunker','eval text chunk','LLM chunk','openai','gemini'],
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Framework :: Jupyter",
        "Operating System :: MacOS",
        "Operating System :: Microsoft",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Microsoft :: Windows :: Windows 11",
        "Operating System :: Unix"
    ],
    cmdclass={"upload": UploadCommand},
)