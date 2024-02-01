# NVIDIA CUDA CUDNN TENSORRT INSTALLATION FOR WSL

It is assumed you already have installed the WSL package on Windows
If not check the read me on that topic
Keep in mind your configuration of OS- if on WSL probably Ubuntu something. It is going to be needed due to compatibility

Also it is assumed you have a Cuda compatible hardware. you can check if it is so using
[this](https://www.techpowerup.com/download/techpowerup-gpu-z/)


## The official documentation
Found [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) 
and [here too](https://docs.nvidia.com/cuda/wsl-user-guide/index.html?highlight=cudnn%20on%20wsl2#getting-started-with-cuda-on-wsl-2)


## NOTE OF CAUTION
If your intention is installing all the dependencies INCLUDING Tensorrt keep in mind the compatibility issues

#### TensorRt 8.6 GA (which is the latest version for now- Jan 2024)
is compatible on CUDA 12.1 (but not further officially)
and
#### CUDA 12.1 
is compatible to CudNN 8.9.0.131 (but not further) 

So what you want to install is 
#### CUDA 12.1
#### CudNN 8.9.0.131
[support matrix here](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)
#### TensorRt 8.6 GA
[support matrix here](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/support-matrix/index.html)

what I ended up installing however was Cuda 12.2, CudNN 8.9.0.131 and Tensorrt 8.6 GA

#
## THE INSTALLATION



1. Start with the Cuda from [here](https://developer.nvidia.com/cuda-toolkit-archive) if you want a different config from the latest 
or if you just want to go with 12.1 [try this one](https://developer.nvidia.com/cuda-12-1-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

2. `sudo apt install nvidia-cuda-toolkit`
will install the cuda compiler in the WSL terminal.
Basically since you have a cuda-enabled hardware you have the drivers required for windows. You don't have all dependencies for Ubuntu however. The command above is used after the cuda instalation from above so that `nvcc --version` returns info

3. Before continuing to cudNN use 

    `apt-cache policy libcudnn8`

    in Linux terminal to check the compatibility as explained [here](https://forums.developer.nvidia.com/t/e-version-8-3-1-22-1-cuda10-2-for-libcudnn8-was-not-found/200801/23)

4. Download the version required from [here](https://developer.nvidia.com/rdp/cudnn-archive)
NOTE:
If at step 2. you couldn't find the version required just check with the release dates by compatibility
And if you are wondering what is the difference between same OS, same cudnn version but different endings in the files (like aarch64sbsa vs cross-sbsa vs none)- it is related to the architecture of your GPU- just go with the simple version

5. After having found your version you should be able to follow the instructions from [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

6. Finally getting to the TensorRT. You should have everything you need [here](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#gettingstarted) 



However- if it can't work by what it is said in those links- do the following
    
6.1 Download the version of TensorRt from [here](https://developer.nvidia.com/nvidia-tensorrt-8x-download) 
    
6.2 Use the same command used in cudNN to install from the file downloaded
`sudo dpkg -i`  
and the name of the archive
   
6.3 don't forget to use
`sudo apt-get install tensorrt`

#
## TROUBLESHOOTING

If by chance your config is not supported you have to backtrack installation and uninstall things
In the order you installed them because there are dependencies at work

You CANNOT just install various other versions- you have to clean install whatever you need- and that means cleaning what has to be cleaned an reinstalling

keep in mind-these commands were very useful

`dpkg -l | grep` 
followed by what you want to search such as cudnn or cuda
it will list everything there is with that name and what you have to uninstall

`sudo dpkg --remove`
or
`sudo dpkg --purge`
followed by the name of the archive you have to uninstall

