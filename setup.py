from setuptools import setup

setup(
    name="choice_model",
    version="0.1",
    packages=["choice_model"],
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
