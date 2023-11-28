from setuptools import find_packages, setup

setup(
    name="coral",
    version="0.2.0",
    description="Package for Coordinate-based network for OpeRAtor Learning",
    author="Louis Serrano",
    author_email="louis.serrano@isir.upmc.fr",
    install_requires=[
        "einops",
        "hydra-core",
        "wandb",
        "torch",
        "pandas",
        "matplotlib",
        "xarray",
        "scipy",
        "h5py",
        "timm",
        "torchdiffeq",
    ],
    package_dir={"coral": "coral"},
    packages=find_packages(),
)
