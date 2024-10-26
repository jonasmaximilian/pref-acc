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
    description="A short description of your project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/prefacc",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)