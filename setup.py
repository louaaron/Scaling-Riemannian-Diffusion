import setuptools

setuptools.setup(
    name="riemdiffpp",
    version="0.0.1",
    author="Aaron Lou",
    author_email="aaronlou@stanford.edu",
    packages=setuptools.find_packages(),
    install_requires=["torch>=1.13.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)