import os.path
from setuptools import setup

from npaes import __version__

setup(
    name="npaes",
    version=__version__,
    author="Brad Solomon",
    author_email="brad.solomon.1124@gmail.com",
    description="Advanced Encryption Standard (AES) NumPy implementation",
    license="Apache 2.0",
    keywords=[
        "aes", "aes-128", "aes-192", "aes-256", "aes 128", "aes 192",
        "aes 256", "encryption", "decryption", "numpy", "symmetric", "cipher"
    ],
    url="https://github.com/bsolomon1124/npaes",
    packages=["npaes"],
    long_description=open(
        os.path.join(os.path.abspath(os.path.dirnaame(__file__)), "README.md")
    ).read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Security :: Cryptography",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    install_requires=["numpy<=1.16,>=1.14.6", "setuptools"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
