========
Welcome to laser-measles
========

.. start-badges

.. image:: https://img.shields.io/pypi/v/laser-measles.svg
    :alt: PyPI Package latest release
    :target: https://test.pypi.org/project/laser-measles/

.. image:: https://img.shields.io/pypi/l/laser-measles.svg
    :alt: MIT License
    :target: https://github.com/InstituteforDiseaseModeling/laser-measles/blob/main/LICENSE    

.. image:: https://readthedocs.org/projects/laser-measles/badge/?style=flat
    :alt: Documentation Status    
    :target: https://laser-measles.readthedocs.io/en/latest/

.. image:: https://codecov.io/gh/InstituteforDiseaseModeling/laser-measles/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/InstituteforDiseaseModeling/laser-measles


.. end-badges

Laser-measles helps you build and analyze spatial models of measles implemented with the LASER toolkit.

Installation
============

.. code-block:: bash

    pip install laser-measles

You can also install the in-development version with:

.. code-block:: bash

    pip install git+https://github.com/InstituteforDiseaseModeling/laser-measles.git@main

Documentation
=============


https://laser-measles.readthedocs.io/en/latest/


Development
===========

To run all the tests run:

.. code-block:: bash

    tox

And to build the documentation run:

.. code-block:: bash

    tox -e docs

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
