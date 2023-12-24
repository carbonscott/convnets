import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="convnets",
    version="23.12.22",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="Build maintainable ConvNets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/convnets",
    keywords = ['Convolutional Neural Nets'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
