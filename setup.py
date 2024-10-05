from setuptools import setup, find_packages

required = [line.strip() for line in open("requirements.txt")]

setup(
    name="rauda_inferencer",
    version="0.0.1",
    packages=find_packages(),
    install_requires=required,
    author="Pablo Merino (Rauda AI)",
    description="A simple inferencing engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rauda-ai/rauda_inferencer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
