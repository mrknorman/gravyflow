from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import subprocess
import sys

class CustomInstall(install):
    """Custom install script to encapsulate environment setup and CUDA compilation."""

    def run(self):
        """Extend the run method to include custom setup steps."""
        # Standard installation process
        install.run(self)

        # Post-installation steps with enhanced feedback and error handling
        try:
            self.set_cuda_env_vars()
            self.compile_cuda_code()
            self.check_and_advise_on_env_persistence()
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please check the installation logs and documentation for guidance.")
            sys.exit(1)

    def set_cuda_env_vars(self):
        """Set CUDA-related environment variables with error handling."""
        try:
            cudnn_path = subprocess.check_output(
                ["python", "-c", "import nvidia.cudnn; print(nvidia.cudnn.__file__)"],
                universal_newlines=True
            ).strip()
            ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
            if "/usr/local/cuda/lib64" not in ld_library_path:
                os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{ld_library_path}"
            if cudnn_path not in ld_library_path:
                os.environ["LD_LIBRARY_PATH"] = f"{os.environ['LD_LIBRARY_PATH']}:{os.path.dirname(cudnn_path)}/lib"
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to set CUDA environment variables: {e}")

    def compile_cuda_code(self):
        """Compile CUDA code if necessary, with error handling."""
        print("Compiling CUDA code...")
        try:
            cuda_dir = os.path.join(os.path.dirname(__file__), "cuphenom")
            os.chdir(cuda_dir)
            subprocess.check_call(["make", "shared"])
            print("CUDA code compiled successfully.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error compiling CUDA code: {e}")
        finally:
            os.chdir(os.path.dirname(__file__))

    def check_and_advise_on_env_persistence(self):
        """Advise the user on making environment variable changes persistent."""
        print("\nPlease ensure CUDA-related environment variables are set persistently.")
        print("This might involve adding them to your shell profile (e.g., .bashrc or .bash_profile).")

setup(
    name="gravyflow",
    version="1.0.0",
    author="Michael Norman",
    author_email="mrknorman@proton.me",
    description="TensorFlow tools for gravitational wave data science.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/mrknorman/gravyflow",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.10',
    install_requires=[
        "gwdatafind",
        "gwpy",
        "pycbc",
        "pytest",
        "GitPython",
        "sqlalchemy",
        "bokeh",
        "psycopg2",
        "ipykernel",
        "tensorflow[and-cuda]",
        "tensorflow-probability",
        "tf-keras"
    ],
    cmdclass={
        'install': CustomInstall,
    },
)
