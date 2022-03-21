# gps_frames
Reference frame representation, transformations, and operations for GPS.

[![Test Python package](https://github.com/the-aerospace-corporation/gps_frames/actions/workflows/python-package.yml/badge.svg)](https://github.com/the-aerospace-corporation/gps_frames/actions/workflows/python-package.yml)
[![CodeQL](https://github.com/the-aerospace-corporation/gps_frames/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/the-aerospace-corporation/gps_frames/actions/workflows/codeql-analysis.yml)

## Installation
This module can be installed using PyPI:
```
pip install gps-frames
```

## Running Tests
This module includes tests for all of the major functionality. To run the tests, you can use the commands in the makefile `make test`. Because some of the functions use JIT compilation via Numba, `make test-nojit` runs all of the tests without JIT compilation to enable better code coverage analysis.

## Using gps_frames
To motivating use case of this module is it determine distances between two points in space while accounting for non-inertial reference frames and non-simultaneous position measures. The basic functionality is demonstrated in `example.py`.ma

Note: When first run, gps-frames has significant overhead due to JIT compliation. This should only occur on the first run. Additionally, you may see `NumbaPerformanceWarning` messages related to the `@` (matrix multiplication) operator. These can be disregarded and should only appear the first time gps-frames is run.

## Licence
The `gps_frames` module is released under the GNU AGPL v3 license.

Copyright (c) 2022 The Aerospace Corportation.

## Open Source Licenses

### EGM96 Data Source
This module makes use of data related to the EGM96 gravity model. This data was generated by the National Geospatial-intelligence Agency (NGA) and the data used is derived from https://github.com/vectorstofinal/geoid_heights, used under the MIT License Copyright (c) 2015 vectorstofinal.

### pdoc3
The documentation is generated using [pdoc3](https://pdoc3.github.io/pdoc/). The documentation templates are based on the default template provided therefrom and used under the AGPL v3 license.
