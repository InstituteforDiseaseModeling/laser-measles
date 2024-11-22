========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |github-actions| |codecov|
    * - package
      - |version| |wheel| |supported-versions| |supported-implementations| |commits-since|
.. |docs| image:: https://readthedocs.org/projects/laser-measles/badge/?style=flat
    :target: https://laser-measles.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/InstituteforDiseaseModeling/laser-measles/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/InstituteforDiseaseModeling/laser-measles/actions

.. |codecov| image:: https://codecov.io/gh/InstituteforDiseaseModeling/laser-measles/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/InstituteforDiseaseModeling/laser-measles

.. |version| image:: https://img.shields.io/pypi/v/laser-measles.svg
    :alt: PyPI Package latest release
    :target: https://test.pypi.org/project/laser-measles/

.. |wheel| image:: https://img.shields.io/pypi/wheel/laser-measles.svg
    :alt: PyPI Wheel
    :target: https://test.pypi.org/project/laser-measles/

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/laser-measles.svg
    :alt: Supported versions
    :target: https://test.pypi.org/project/laser-measles/

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/laser-measles.svg
    :alt: Supported implementations
    :target: https://test.pypi.org/project/laser-measles/

.. |commits-since| image:: https://img.shields.io/github/commits-since/InstituteforDiseaseModeling/laser-measles/v0.6.0.svg
    :alt: Commits since latest release
    :target: https://github.com/InstituteforDiseaseModeling/laser-measles/compare/v0.6.0...main



.. end-badges

Spatial models of measles implemented with the LASER toolkit.

* Free software: MIT license

Installation
============

::

    pip install laser-measles

You can also install the in-development version with::

    pip install https://github.com/InstituteforDiseaseModeling/laser-measles/archive/main.zip


Documentation
=============


https://laser-measles.readthedocs.io/en/latest/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
