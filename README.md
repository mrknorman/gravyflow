# GravyFlow

TensorFlow tools to facilitate machine learning for gravitational-wave data analysis. 

# Installation Guide

## 1. Clone the Repository
GravyFlow can be installed by cloning the Git repository:

```bash
git clone https://github.com/mrknorman/gravyflow
```

## 2. Install Gravyflow
It is recommended to install GravyFlow within a new conda environment. GravyFlow requires Python 3.11:

```bash
conda create --name gravyflow python=3.13
conda activate gravyflow
```

Next, ensure pip is installed by running:

```bash
conda install pip 
```

Then, install GravyFlow and its requirements into your conda environment:

```bash
pip install -e .
```

Note that GravyFlow is under active development, and you may encounter issues during installation. Ensure TensorFlow can recognize GPUs in your environment, as GravyFlow is optimized for GPU use and relies on vectorized GPU functions.

## 4. Setup permissions:

Follow these guides for setting up permissions to access real data:

https://computing.docs.ligo.org/guide/auth/scitokens/
https://computing.docs.ligo.org/guide/auth/kerberos/

## 5. Setup Gravity Spy Permissions:

Access Gravity Spy credentials by logging in with your LIGO credentials at:

https://secrets.ligo.org/secrets/144/

Then, export the obtained username and password:

```bash
export GRAVITYSPY_DATABASE_USER=<user>
export GRAVITYSPY_DATABASE_PASSWD=<password>
```

## 6. Test Gravyflow (optional)

GravyFlow includes PyTest for testing its functionality. To run tests:

```bash
pytest gravyflow
```

Note: Tests may fail due to unavailable GPU memory if GPUs are currently under heavy use.
