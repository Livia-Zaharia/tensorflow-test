# REBUILDING FROM SOURCE OF TENSORFLOW

First of all keep in mind that after Tensorflow 2.10 all Tensorflow to run with GPU runs on Linux
This Read me is based on instruction from [here](https://www.tensorflow.org/install/source_windows) but with some additional notes.
To compile the necessary Tensorflow for Gpu enabling you must

1. if on windows install WSL as found [here](https://learn.microsoft.com/en-us/windows/wsl/install)

From here on the steps are all in Linux/WSL unless specified otherwise


2. install python and dependencies in Linux as shown [here](https://phoenixnap.com/kb/how-to-install-python-3-ubuntu)


3. on the terminal in Linux install Bazel as shown [here](https://bazel.build/install/ubuntu)


4. in VS code install bazel and wsl extensions


5. MYSYS2 installation from [here] (https://www.msys2.org/wiki/MSYS2-installation/) and [here] (https://repo.msys2.org/distrib/x86_64/)


6. Don't forget to install the Microsoft Visual C++ 2019 Redistributable and
Microsoft Build Tools 2019. See [here] (https://visualstudio.microsoft.com/downloads/) for more


7. For reference reasons check your OS version with
`lsb_release -a` 


8. check compatibility of versions [here](https://www.tensorflow.org/install/source#gpu)


9. follow INSTALLING NVIDIA CUDA_CUDNN_TENSORT_FOR  WSL read-me instructions
Just in case you had previous versions installed and there are still there you can use this command to set the current version of cuda (symlinking)
`sudo ln -s /usr/local/cuda-<version> /usr/local/cuda`


10. install Clang from [here](https://apt.llvm.org/) if you need 16 or above- 
or just use
`sudo apt install clang <version>` if bellow or equal to 15

`sudo ln -s /usr/bin/clang-16 /usr/bin/clang` just in case you have another one in before

Be sure to have all the libraries in llvm. It's important!


11. It is recommended to have pyenv installed on WSL.To run another version of Python under WSL, you can use a version manager such as pyenv. This tool allows you to easily switch between multiple versions of Python.


Install Dependencies first
`sudo apt-get install --yes git libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libgdbm-dev lzma lzma-dev tcl-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev wget curl make build-essential  python-openssl `


Clone the pyenv Repository
`git clone https://github.com/pyenv/pyenv.git ~/.pyenv`


After cloning the pyenv repository, you need to add pyenv to your bash configuration file. First open the the bash configuration file
` nano ~/.bashrc.` or `nano ~/.bash_profile`

In the opened file, add the following lines:
`  
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    if command -v pyenv 1>/dev/null 2>&1; then
    eval "$(pyenv init -)"
    fi 
`
These lines initialize pyenv and add it to your PATH.

Save your changes and close the file. You can do this by pressing Ctrl+X, then Y, then Enter.
Finally, reload your shell profile so the changes take effect. You can do this by closing and reopening your terminal, or by sourcing the profile file with the following command:
`source ~/.bash_profile`

Now, you can install a specific version of Python using pyenv. For example, to install Python 3.8.0, you can use the following command:
`pyenv install 3.8.0`

In our case 3.9.0, 3.10.0, 3.11.0 would be recommended.
After installing a Python version, you can set it as the global Python version with the following command:
`pyenv global 3.8.0`


12. install dos2unix
`sudo apt install dos2unix`


13. download source for Tensorflow from git as seen [here](https://github.com/tensorflow/tensorflow)


14. To this date (February 2023) the version I built was based on branch r2.15
The following steps are important if you are building on WSL especially or possibly Linux


15. After cloning from git if you are running on WSL the file was copied most likely on the Windows partition.
Create a .gitattributes file with the following content
`* text=auto eol=lfpython3`
That file will configure the files updated through checkout to have an Unix style EOL
For more details read [here](https://www.phind.com/search?cache=nre288el69152ptrvgzsicmo)


16. `git checkout r2.15` would work faster if run in normal terminal (powershell or whichever Windows based). That is why we added the .gitattributes file. However due to the formatting of the source code there are EOL windows formatted in the code itself. So after the checkout in powershell we must also use dos2unix to convert the rest, run, however in the ubuntu terminal. 
`find /path/to/directory -type f -exec dos2unix {} \;`


17. `pyenv global 3.10`


18. still in the ubuntu terminal in VS code and run configure.py
`python3 configure.py`

        python3 configure.py
        You have bazel 6.1.0 installed.
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
        Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 8.9]: compute_89


        Do you want to use clang as CUDA compiler? [Y/n]: y
        Clang will be used as CUDA compiler.

        Please specify clang path that to be used as host compiler. [Default is /usr/lib/llvm-16/bin/clang]:


        You have Clang 16.0.6 installed.

        Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -Wno-sign-compare]:


        Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n     
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



19. the command to build the library has the -xclang -fcuda-allow variadic-functions because [this](https://github.com/tensorflow/tensorflow/issues/62339)
` /usr/bin/bazel build --verbose_failures --config=opt --config=cuda --define=no_tensorflow_py_deps=true --copt=-Xclang --copt=-fcuda-allow-variadic-functions //tensorflow/tools/pip_package:build_pip_package`

20. until  now  we have built a package-builder but we have to build the pip package so first we need patchelf

`sudo apt-get install patchelf` because [otherwise](https://www.phind.com/search?cache=vs5bl8zy0t7ni73v46nj9vzr)

21. `./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg`
Builds the whl package
Important! location in temporary folder- better to copy the package to another location.

22. keep in mind- this line is if you have the whl at that adress.
 `pip install /tmp/tensorflow_pkg/tensorflow-version-tags.whl`

 also if you don't want to install it globally you could install it in venv.

[Some details about what goes on between bazel- clang and tensorflow](https://www.phind.com/search?cache=pisgfraz05o6kly0449pn1v4)
[Some reading on the process- pretty good even though it is an older version](https://medium.com/analytics-vidhya/building-tensorflow-2-0-with-gpu-support-and-tensorrt-on-ubuntu-18-04-lts-part-2-ff2b1482c0a3)