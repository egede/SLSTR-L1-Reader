from setuptools import setup, find_packages
setup(
    name="SLSTR-L1_Reader",
    version="0.1",
    packages=find_packages(),

    install_requires=['matplotlib', 'numpy', 'netCDF4', 'scipy'],

    # metadata to display on PyPI
    author="Ulrik Egede",
    author_email="u.egede@imperial.ac.uk",
    description="Python interface for reading SLSTR Level 1 information",
    license="Gnu General Public Licence v3",
    keywords="Sentinel Copernicus SLSTR L1 remote sensing",
    url="https://github.com/egede/SLSTR-L1-Reader",
)
