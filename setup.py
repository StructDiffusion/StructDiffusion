import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="StructDiffusion-wliu88",
    version="0.0.1",
    author="Weiyu Liu",
    author_email="wliu88@gatech.edu",
    description="Source code for StructDiffusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wliu88/StructDiffusion",
    project_urls={
        "Bug Tracker": "https://github.com/wliu88/StructDiffusion/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)

