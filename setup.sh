#!/bin/bash

source ~/.bashrc

function set_env_vars() {
    # Set up CUDNN environment variables
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo "CUDNN environment variables set."
}

function reinstall_env() {
    conda deactivate
    mamba env remove -n gravyflow
    mamba env create -f environment.yml
    conda activate gravyflow
    set_env_vars
    echo "Reinstalled the 'gravyflow' environment."
}

function progress_bar() {
    local total=$1
    local current=$2
    local width=50  # Width of the progress bar
    local perc=$((($current * 100) / $total))
    local done=$((($perc * $width) / 100))
    local remaining=$(($width - $done))
    printf "\rProgress: ["
    printf "%0.s#" $(seq 1 $done)
    printf "%0.s-" $(seq 1 $remaining)
    printf "] %d%%" $perc
}

function gravyflow_cuphenom_install() {
    # Navigate to gravyflow/cuphenom directory
    cd cuphenom

    # Set environment variables for CUDA
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}"

    # Compile shared objects using make
    make shared

    # Navigate back to the original directory
    cd -
}

# Try creating the conda environment and capture any errors
mamba_error=$(mamba env create -f environment.yml 2>&1)

# Check if the error about existing prefix is in the captured errors
if [[ $mamba_error == *"CondaValueError: prefix already exists"* ]]; then
    echo "NOTICE: The 'gravyflow' environment already exists. Checking its integrity..."
    
else
    echo "$mamba_error."
fi

# Activate the environment
conda activate gravyflow

# Extracting just the package names
required_packages=$(awk '/^dependencies:/ {flag=1; next} /^  -/ && flag && !/pip:/ {print $0} /^[^ ]/ && flag {flag=0}' environment.yml | sed 's/^  - //' | cut -d'=' -f1)

# Verify that all packages in environment.yml are in the environment
total_packages=$(echo "$required_packages" | wc -l)
missing_packages=""
count=0

echo "Checking packages..."
for pkg in $required_packages; do
    count=$((count + 1))
    progress_bar $total_packages $count

    # Using conda list to check if package (by name) is installed in the environment
    if ! conda list -n gravyflow | grep -qw "^$pkg[[:space:]]"; then
        missing_packages+="$pkg "
    fi
done
printf "\n"  # Move to a new line after the progress bar

# Print missing packages, if any
if [[ -n $missing_packages ]]; then
    echo "The following packages are missing: $missing_packages"
fi

# Verify environment variables
env_vars_ok=true
if [[ -z "$LD_LIBRARY_PATH" || ! "$LD_LIBRARY_PATH" == *"$CONDA_PREFIX/lib/"* ]]; then
    env_vars_ok=false
    set_env_vars  # Reset the environment variables if they're not set correctly
fi

# Test TensorFlow GPU detection
output=$(python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))")
gpu_detected=true
if [[ ! $output == *'GPU'* ]]; then
    gpu_detected=false
fi

# Decide on reinstall based on issues
issues_detected=false
issue_message="WARNING: Issues detected. Reasons:"

if [[ -n $missing_packages ]]; then
    issues_detected=true
    issue_message="$issue_message\n- Missing packages: $missing_packages"
fi

if [[ $env_vars_ok == false ]]; then
    issues_detected=true
    issue_message="$issue_message\n- Environment variables not correctly set."
fi

if [[ $gpu_detected == false ]]; then
    issues_detected=true
    issue_message="$issue_message\n- No GPUs detected."
fi

if [[ $issues_detected == true ]]; then
    echo -e $issue_message
    echo "Reinstalling environment..."
    reinstall_env
fi

if [[ $gpu_detected == false ]]; then
    echo "WARNING: No GPUs detected even after reinstall! GravyFlow requires GPUs for correct function."
else
    echo "GPUs detected: $output"

    echo "Compiling cuPhenom..."
    gravyflow_cuphenom_install
fi