#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    setuptools.setup()


# import os
# import platform
# import re
# from pathlib import Path

# from setuptools import Extension
# from setuptools import find_packages
# from setuptools import setup
# from setuptools.dist import Distribution

# # Enable code coverage for C code: we cannot use CFLAGS=-coverage in tox.ini, since that may mess with compiling
# # dependencies (e.g. numpy). Therefore, we set SETUPPY_CFLAGS=-coverage in tox.ini and copy it to CFLAGS here (after
# # deps have been safely installed).
# if "TOX_ENV_NAME" in os.environ and os.environ.get("SETUPPY_EXT_COVERAGE") == "yes" and platform.system() == "Linux":
#     CFLAGS = os.environ["CFLAGS"] = "-fprofile-arcs -ftest-coverage"
#     LFLAGS = os.environ["LFLAGS"] = "-lgcov"
# else:
#     CFLAGS = ""
#     LFLAGS = ""


# class BinaryDistribution(Distribution):
#     """
#     Distribution which almost always forces a binary package with platform name
#     """

#     def has_ext_modules(self):
#         return super().has_ext_modules() or not os.environ.get("SETUPPY_ALLOW_PURE")


# def read(*names, **kwargs):
#     with Path(__file__).parent.joinpath(*names).open(encoding=kwargs.get("encoding", "utf8")) as fh:
#         return fh.read()


# setup(
#     long_description="{}\n{}".format(
#         re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub("", read("README.rst")),
#         re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
#     ),
#     url="https://github.com/InstituteforDiseaseModeling/laser-measles",
#     packages=find_packages("src"),
#     package_dir={"": "src"},
#     py_modules=[path.stem for path in Path("src").glob("*.py")],
#     include_package_data=True,
#     zip_safe=False,
#     ext_modules=[
#         Extension(
#             str(path.relative_to("src").with_suffix("")).replace(os.sep, "."),
#             sources=[str(path)],
#             extra_compile_args=CFLAGS.split(),
#             extra_link_args=LFLAGS.split(),
#             include_dirs=[str(path.parent)],
#         )
#         for path in Path("src").glob("**/*.c")
#     ],
#     distclass=BinaryDistribution,
# )
