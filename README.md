# Getting Started

## Installation

- Cuda
- CuDNN
- scipy
- numpy
- Tensorflow
- Keras
- Pandas
- H5py

## Install dependencies

1. Update apt-get

   ```
    sudo apt-get update
   ```

2. Install dependencies

   ```
    sudo apt-get install libglu1-mesa libxi-dev libxmu-dev -y
    sudo apt-get install build-essential -y
    sudo apt-get install python3-pip python3-dev -y
    sudo apt-get install python3-numpy python3-scipy -y
    sudo apt-get install python3-virtualenv virtualenv -y
    sudo apt-get install python-yaml -y
    sudo apt-get install libhdf5-serial-dev -y
   ```

## Install CUDA

Cuda is architecture platform supporting parallel computing created by Nvidia. It is essentially a software layer between the CPU and the GPU. Cuda is well-suited for both graphical applications like 2D or 3D modelling, graphic intensive video games as well as computational applications in biology or in crypography. Interestingly, machine learning applications can be both graphically as well as computationally intensive. Hence, every library or framework that we are going to talk about comes with cuda support. The cuda toolkit has inbuilt libraries performing important computations involving linear algebra, fast fourier transforms etc. Cuda makes single threaded workflow on CPU and accelarated parallel processing on GPU possible. The cuda execution model has three parts- grids, blocks and threads.The grid runs on a device(GPU), blocks run on multi processors and threads run on scalar processor(cores). With cuda, hardware resource allotment for thousands of threads running in hundreds or thousands of GPU cores over millions or even billions of transistors is made extremely efficient. Cuda essentially splits up code into threads by itself and assigns them to GPU cores thus highly speeding up computing.

### Deb file method

1. Check if you have nvidia graphic card by running the following command in the terminal

   ```
    lspci -nnk | grep -i nvidia
   ```

   If there is no output then you can skip to installing scipy.

2. Goto [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) and download the deb (local) file for your system. To check distribution run` uname -m && cat /etc/*release`

3. Install repository meta-data

   ```
    sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
   ```

   Update apt repository cache

   ```
    sudo apt-get update
   ```

   Install CUDA

   ```
    sudo apt-get install cuda
   ```

4. Open .bashrc

   ```
    nano ~/.bashrc
   ```

   Add the following lines to the end of the file

   ```
    export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
   ```

   Exit out of nano by pressing Ctrl+x then y and then enter Now run the following command

   ```
    source ~/.bashrc
   ```

5. Verify driver installation

   ```
    cat /proc/driver/nvidia/version
   ```

   Verify cuda installation

   ```
    nvcc -V
   ```

### Run file method

Use this method if the deb file method doesn't work.

1. Check if you have nvidia graphic card by running the following command in the terminal

   ```
    lspci -nnk | grep -i nvidia
   ```

   If there is no output then you can skip to installing scipy.

2. Goto [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) and download the run file for your system.

3. Disable Nouveau drivers

   ```
    sudo nano /etc/modprobe.d/blacklist-nouveau.conf
   ```

   Write the following inside the file and press Ctrl+x and then Enter

   ```
    blacklist nouveau 
    options nouveau modeset=0
   ```

   Run the following in terminal to update initial RAM filesystem

   ```
    sudo update-initramfs -u
   ```

4. Run the following command to make run file executable

   ```
    sudo chmod +x cuda_8.0_linux.run
   ```

5. Reboot Ubuntu and press Alt+F1 when prompted for login

6. Purge any nvidia packages

   ```
    sudo apt-get remove --purge nvidia*
    sudo apt-get autoremove
   ```

7. Run the following command to install CUDA

   ```
    sudo ./cuda_8.0_linux.run --override
   ```

8. Open .bashrc

   ```
    nano ~/.bashrc
   ```

   Add the following lines to the end of the file

   ```
    #CUDA Toolkit
    export CUDA_HOME=/usr/local/cuda-8.0
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
    export PATH=${CUDA_HOME}/bin:${PATH}
   ```

   Exit out of nano by pressing Ctrl+x then y and then enter Now run the following command and reboot

   ```
    source ~/.bashrc
    sudo reboot
   ```

9. Verify driver installation

   ```
    cat /proc/driver/nvidia/version
   ```

   Verify cuda installation

   ```
    nvcc -V
   ```

## Install cuDNN

The cuDNN library provides highly tuned implemntation of standard deep learning routines such as convolution, poolinng, normalization, activation, optimization. CuDNN accelaration GPU performance for deep learning frameworks like Tensorflow , Theano, Caffe etc.

1. Register for a nvidia developer account and head over to [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) and download cuDNN v5.1 for CUDA 8.0

2. Unpack the file

   ```
    tar -xzvf cudnn-8.0-linux-x64-v5.1.tgz
   ```

   Copy cuDNN files to cuda folder

   ```
    sudo cp cuda/lib64/* /usr/local/cuda/lib64/
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
   ```

## Set up Virtualenv

1. Create virtual environment

   ```
    virtualenv --system-site-packages ~/kerai
   ```

2. Activate virtual environment, this is what you need to run every time you want to work with tensorflow and keras

   ```
    source ~/kerai/bin/activate
   ```

   Your terminal prompt should change to

   ```
    (kerai)$
   ```

   To deactivate the virtual environment execute

   ```
    deactivate
   ```

## Install tensorflow

Tensorflow is an open source library for end-to end machine learning apllications. It was developed by Google's Brain team and is written in python, C++, cuda. It is based on stateful dataflow graphs distributed over tensorflow sessions using multidimensional arrays called tensors. Tensorflow has inbuilt tensorboard for visualizing the graphs.

- If you do not have a Nvidia graphics card the run the following command

  ```
    (kerai)$ pip3 install --upgrade tensorflow
  ```


- If you have Nvidia graphic card and installed CUDA and cuDNN then run

  ```
    (kerai)$ pip3 install --upgrade tensorflow-gpu
  ```

## Install Keras

Keras is a essentially a model based library written in native python for building and deploying neural networks. It was too developed by a Google engineer. It can run over Theano or Tensorflow backend. The key feature of keras is its **readability** and **modularity**without diving into tensorflow or theano.

- Install keras in the virtualenv

  ```
    (kerai)$ pip3 install keras
  ```

## Install optional libraries

- Pandas

  ```
    (kerai)$ pip3 install pandas
  ```


- H5py to save models

  ```
    (kerai)$ pip3 install h5py
  ```