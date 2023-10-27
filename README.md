# GravyFlow
TensorFlow tools to facilitate machine learning for gravitational-wave data analysis. 

# Install guide:

1. Clone the git repo, ensuring you recurse submodules:
```
git clone --recurse-submodules https://github.com/mrknorman/gravyflow.git
```

2. If not installed, install mamba for faster environment solving:
```
conda install mamba -c conda-forge
```

3. Navigate to GravyFlow directory:
```
cd gravyflow
```

4. Give the install script permissions:
```
chmod +x setup.sh
```

5. Run the install script:
```
./setup.sh
```

6. To activate the enviroment use:
```
conda activate gravyflow
```

7. Setup permissions:

Follow this guide: 

https://computing.docs.ligo.org/guide/auth/x509/

Then this guide:

https://computing.docs.ligo.org/guide/auth/kerberos/

