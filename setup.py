import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="docbank-loader", # Replace with your own username
    version="1.0.0",
    author="Minghao Li",
    author_email="li_mh1997@163.com",
    description="A dataset loader for DocBank",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/doc-analysis/DocBank",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)