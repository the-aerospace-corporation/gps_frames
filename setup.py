# Copyright (c) 2022 The Aerospace Corporation
from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="gps_frames",
    version="2.8.0",
    description="Reference frames representations and transformations",
    url="https://github.com/the-aerospace-corporation/gps_frames",  # noqa: E501
    author="David William Allen",
    author_email="david.w.allen@aero.org",
    license="GNU AGPL v3",
    packages=[
        "gps_frames",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7, <3.10",
    install_requires=required,
    include_package_data=True,
    zip_safe=False,
)
