# cuphenom

Package to generate PhenomIMR waveforms using NVidia's CUDA GPU libary.

# Compile CuPhenom:

On the LIGO cluster CuPhenom should be compilable with little difficulty. Run these commands to install:

```
cd cuphenom
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}$
make shared
```
