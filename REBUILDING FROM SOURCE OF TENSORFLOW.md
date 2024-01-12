# REBUILDING FROM SOURCE OF TENSORFLOW

To compile the necessary Tensorflow for Gpu enabling you must

1. if on windows install WSL as found [here](https://learn.microsoft.com/en-us/windows/wsl/install)

From here on the steps are all in Linux/WSL

2. install python and dependencies in Linux as shown [here](https://phoenixnap.com/kb/how-to-install-python-3-ubuntu)

3. on the terminal in Linux install Bazel as shown [here](https://bazel.build/install/ubuntu)

4. in VS code install bazel and wsl extensions

5. For reference reasons check your OS version with
`lsb_release -a` 

6. follow INSTALLING NVIDIA CUDA_CUDNN_TENSORT_FOR  WSL read-me instructions

7. download source for Tensorflow from git as seen [here](https://github.com/tensorflow/tensorflow)

8. open the ubuntu terminal in VS code and run configure.py

9.   
    python3 configure.py
    WARNING: current bazel installation is not a release version.
    Please specify the location of python. [Default is /usr/bin/python3]:


    Found possible Python library paths:
      /usr/lib/python3/dist-packages
      /usr/local/lib/python3.10/dist-packages
    Please input the desired Python library path to use.  Default is [/usr/lib/python3/dist-packages]        

    Do you wish to build TensorFlow with ROCm support? [y/N]: n
    No ROCm support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with CUDA support? [y/N]: y
    CUDA support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with TensorRT support? [y/N]: y
    TensorRT support will be enabled for TensorFlow.

    Found CUDA 12.1 in:
        /usr/local/cuda-12.1/targets/x86_64-linux/lib
        /usr/local/cuda-12.1/targets/x86_64-linux/include
    Found cuDNN 8 in:
        /usr/lib/x86_64-linux-gnu
        /usr/include
    Found TensorRT 8.6.1 in:
        /usr/lib/x86_64-linux-gnu
        /usr/include/x86_64-linux-gnu


    Please specify a list of comma-separated CUDA compute capabilities you want to build with.
    You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus. Each capability can be specified as "x.y" or "compute_xy" to include both virtual and binary GPU code, or as "sm_xy" to only include the binary code.
    Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 8.9]: 8.9


    Do you want to use clang as CUDA compiler? [Y/n]: n
    nvcc will be used as CUDA compiler.

    Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:


    Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -Wno-sign-compare]:


    Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:
    Not configuring the WORKSPACE for Android builds.

    Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
            --config=mkl            # Build with MKL support.
            --config=mkl_aarch64    # Build with oneDNN and Compute Library for the Arm Architecture (ACL).  
            --config=monolithic     # Config for mostly static monolithic build.
            --config=numa           # Build with NUMA support.
            --config=dynamic_kernels        # (Experimental) Build kernels into separate shared objects.     
            --config=v1             # Build with TensorFlow 1 API instead of TF 2 API.
    Preconfigured Bazel build configs to DISABLE default on features:
            --config=nogcp          # Disable GCP support.
            --config=nonccl         # Disable NVIDIA NCCL support.