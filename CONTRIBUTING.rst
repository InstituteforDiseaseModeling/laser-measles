============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

Bug reports
===========

When `reporting a bug <https://github.com/InstituteforDiseaseModeling/laser-measles/issues>`_ please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.

Documentation improvements
==========================

laser_measles could always use more documentation, whether as part of the
official laser_measles docs, in docstrings, or even on the web in blog posts,
articles, and such.

Feature requests and feedback
=============================

The best way to send feedback is to file an issue at https://github.com/InstituteforDiseaseModeling/laser-measles/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that code contributions are welcome :)

Development
===========

To set up `laser-measles` for local development:

1. Fork `laser-measles <https://github.com/InstituteforDiseaseModeling/laser-measles>`_
   (look for the "Fork" button).
2. Clone your fork locally::

    git clone git@github.com:YOURGITHUBNAME/laser-measles.git

3. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

4. When you're done making changes run all the checks and docs builder with one command::

    tox

5. Commit your changes and push your branch to GitHub::

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

6. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests (run ``tox``).
2. Update documentation when there's new API, functionality etc.
3. Add a note to ``CHANGELOG.rst`` about the changes.
4. Add yourself to ``AUTHORS.rst``.

Tips
----

To run individual tox environments::

    # Run code quality checks (linting, formatting, etc.)
    tox -e check

    # Run tests (including slow tests)
    tox -e py312

    # Run tests excluding slow tests (same as CI)
    tox -e py312-ci

    # Build documentation
    tox -e docs

To run a subset of tests::

    tox -e envname -- pytest -k test_myfeature

To run all the test environments in *parallel*::

    tox -p auto

Version Management
==================

This project uses `bump-my-version <https://github.com/callowayproject/bump-my-version>`_ for automated version management. The tool is configured in ``pyproject.toml`` and automatically updates version numbers across multiple files.

Common version bump commands::

    # Patch release (bug fixes): 0.7.0 -> 0.7.1
    bump-my-version bump patch

    # Minor release (new features): 0.7.0 -> 0.8.0
    bump-my-version bump minor

    # Major release (breaking changes): 0.7.0 -> 1.0.0
    bump-my-version bump major

    # Development version: 0.7.0 -> 0.7.1-dev0
    bump-my-version bump patch --new-version 0.7.1-dev0

    # Release candidate: 0.7.1-dev0 -> 0.7.1-rc0
    bump-my-version bump pre_l --new-version 0.7.1-rc0

The tool automatically updates version numbers in:

* ``pyproject.toml``
* ``setup.py``
* ``docs/conf.py``
* ``src/laser_measles/__init__.py``
* ``.cookiecutterrc``

By default, bump-my-version will create a git commit and tag. The configuration supports semantic versioning with pre-release labels (dev, rc, final).
