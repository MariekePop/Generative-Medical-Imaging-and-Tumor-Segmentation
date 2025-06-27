from setuptools import setup, find_packages

setup(
    name="nnunetv2",
    version="0.0.1",
    packages=find_packages(include=["nnunetv2", "nnunetv2.*"]),
    install_requires=[
        "numpy",
        "SimpleITK",
        "medpy",
    ],
)
