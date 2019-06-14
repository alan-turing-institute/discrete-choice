# choice_model [![Build Status](https://travis-ci.com/alan-turing-institute/discrete-choice.svg?branch=master)](https://travis-ci.com/alan-turing-institute/discrete-choice)

A Python package for defining discrete choice models and solving them using
numerous back ends through a single interface.

This module is a result of the [ALOGIT in
Python](https://www.turing.ac.uk/research/research-projects/common-interface-discrete-choice)
project, part of the [Research
Engineering](https://www.turing.ac.uk/research/research-programmes/research-engineering)
programme at the Alan Turing Institute.

## Currently supported back ends

- [ALOGIT](http://www.alogit.com/)
- [Biogeme](http://biogeme.epfl.ch/)
- [pylogit](https://github.com/timothyb0912/pylogit)

## Currently supported models

- Multinomial logit

## Installation

Install the packages and dependencies with `pip install .`

## Testing

The pytest module (`pip install pytest`) is required to run the tests. The tests
can be run with `python -m pytest`.

## Examples

Example validation and benchmarking scripts are given in the examples directory.
