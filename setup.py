from setuptools import setup, find_packages

setup(
    name="choice_model",
    version="0.1",
    packages=find_packages(),
    license="MIT",
    author="Jim Madge",
    url="https://github.com/alan-turing-institute/discrete-choice",
    install_requires=[
        "numpy",
        "pandas",
        "pylogit",
        "pyyaml",
        "scipy"
    ]
)
