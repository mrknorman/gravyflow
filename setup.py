from setuptools import setup, find_packages

setup(
    name="gravyflow",
    version="1.0.0",
    author="Michael Norman",
    author_email="mrknorman@proton.me",
    description="Jax and Keras-3 tools for gravitational wave data science.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/mrknorman/gravyflow",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.13',
    install_requires=[
        "jax[cuda12]",
        "keras>=3.0",
        "ripplegw",
        "gwdatafind",
        "gwpy",
        "pycbc",
        "pytest",
        "pytest-env",
        "GitPython",
        "sqlalchemy",
        "bokeh",
        "psycopg2-binary",
        "ipykernel",
    ],
)
