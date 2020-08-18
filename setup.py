'''
Author: glchen
Date: 2020-07-15 17:34:48
LastEditTime: 2020-08-08 12:49:56
Description: In User Settings Edit
'''
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="glimg",
    version="0.0.10",
    author="glchen",
    author_email="chencglt@gmail.com",
    description="image processing tool for computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chencgln/glimg",
    packages=setuptools.find_packages(),
    classifiers=[]
        # "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: Apache License",
        # "Operating System :: OS Independent",
)





