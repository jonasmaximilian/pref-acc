from setuptools import setup, find_packages

setup(
    name="prefacc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "brax",
        "jax",
        "numpy",
        "absl-py",
        "matplotlib",
        "pandas",
    ],
    author="JSR",
    url="https://github.com/jonasmaximilian/pref-acc",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)