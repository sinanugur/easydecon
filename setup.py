from setuptools import setup, find_packages, Command
import os

with open("requirements.txt") as req:
    requirements=req.readlines()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


with open("README.md","r") as readme:
    long_description=readme.read()

setup(
    name="easydecon",
    version="0.1.1b1",
    packages=find_packages(exclude=('tests*','testing*')),
    #scripts=["scripts/raw_to_segmented_h5ad.py"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    #extras_require={"dev":["pytest>=3.7"]},
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': ['run_bin2cell_segmentation=easydecon.segmentation:main'],
    },
    # metadata to display on PyPI
    author="Sinan U. Umu",
    author_email="sinanugur@gmail.com",
    description="easydecon",
    keywords="scRNA single-cell high definition spatial transcriptomics deconvolution",
        cmdclass={
        'clean': CleanCommand,
    },
    url="https://github.com/sinanugur/easydecon",   # project home page, if any
    classifiers=[
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
        'Topic :: Scientific/Engineering :: Bio-Informatics'

    ]

    # could also include long_description, download_url, etc.
)
