# GravitationalFlow
TensorFlow tools to facilitate machine learning for gravitational-wave data analysis. 

# Environment Setup:

```
conda create -n py_ml_tools_310 python=3.10
conda activate py_ml_tools_310
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
conda install -c conda-forge gwdatafind
conda install -c conda-forge gwpy
conda install tensorflow-probability==0.20.*
conda install -c conda-forge python-lalframe
conda install bokeh
```
# Compile CuPhenom:

On the LIGO cluster CuPhenom should be compilable with little difficulty. Run these commands to install:

```
cd py_ml_tools\cuphenom
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}$
make shared
```


# Setup permissions

Follow this guide: 

https://computing.docs.ligo.org/guide/auth/x509/

And this guide:

https://computing.docs.ligo.org/guide/auth/kerberos/
