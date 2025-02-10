from setuptools import find_packages, setup

setup(
    name="dataset_generator",
    version="0.1.0",
    author="Mikhail Kalinichenko",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/example_package",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "albumentations==2.0.3",
        "opencv-python-headless==4.11.0.86",
        "numpy==2.2.2",
    ],
)
