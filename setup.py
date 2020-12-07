
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

import setuptools


DIR = os.path.dirname(__file__)
REQUIREMENTS = os.path.join(DIR, "requirements.txt")


with open(REQUIREMENTS) as f:
    reqs = f.read()

setuptools.setup(
    name="stl_text",
    version="0.0.1",
    author="Facebook",
    license="BSD",
    packages=setuptools.find_packages(),
    install_requires=reqs.strip().split("\n"),
)