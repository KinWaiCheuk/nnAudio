import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nnAudio", # Replace with your own username
    version="0.1.1",
    author="KinWaiCheuk",
    author_email="u3500684@connect.hku.hk",
    description="A fast GPU audio processing toolbox with 1D convolutional neural network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KinWaiCheuk/nnAudio",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
