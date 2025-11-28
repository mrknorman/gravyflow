# GravyFlow

TensorFlow tools to facilitate machine learning for gravitational-wave data analysis. 

# Installation Guide

## 1. Clone the Repository
GravyFlow can be installed by cloning the Git repository. Execute the following command to clone GravyFlow along with its submodules:

```bash
git clone --recurse-submodules https://github.com/mrknorman/gravyflow
```

The `--recurse-submodules` option ensures that the cuPhenom submodule and its submodules are downloaded.

## 2. Compile cuPhenom

To compile the cuPhenom library, navigate to the cuPhenom submodule directory:

```bash
cd gravyflow/gravyflow/cuphenom
```

Ensure the CUDA C++ compiler (`nvcc`) and associated CUDA libraries are accessible by updating your PATH variable:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}$
```

Adjust the path if your CUDA installation is in a non-default location. Compile cuPhenom by running:

```bash
make shared
```

Afterward, return to the previous directory to continue the installation process:

```bash
cd -
```

## 3. Install Gravyflow
It is recommended to install GravyFlow within a new conda environment. GravyFlow requires Python 3.10:

```bash
conda create --name gravyflow
conda activate gravyflow
```

Next, ensure pip is installed by running:

```bash
conda install pip 
```

Then, install GravyFlow and its requirements into your conda environment:

```bash
pip install -e gravyflow
```

Note that GravyFlow is under active development, and you may encounter issues during installation. Ensure TensorFlow can recognize GPUs in your environment, as GravyFlow is optimized for GPU use and relies on vectorized GPU functions. Some components, especially cuPhenom, do not have CPU fallbacks.

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
