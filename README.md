# GSBN
 
<strong>G</strong>PU version of <strong>S</strong>piking-based <strong>B</strong>CPN<strong>N</strong>.

## Features
1. Support of GPU acceleration (CUDA GPU).
2. Support of MPI multi-processing acceleration.
3. Support of multiple simulation mode, including CPU, GPU and Mix-CPU-GPU.
4. Everything is defined by configuration file.
5. Framework architecture. Procedures are loaded at run-time according to configuration file.

## Compilation
GSBN can be compiled with or without CUDA. If CUDA is not present, the program can only run in CPU mode. Parallelism can be achieved by both GPU and MPI multi-processing.

### Dependencies
To compile GSBN without GPU support, You only need:
 1. Build tools: g++, cmake, etc.
 2. Google Protobuf.
 3. Any MPI program and library: OpenMPI, IntelMPI, etc.

On Ubuntu linux, you can use command below to install all of them.
````
sudo apt-get install build-essential cmake libprotobuf-dev protobuf-compiler libopenmpi-dev openmpi-bin openmpi-doc
````

Extra dependency for compiling GSBN with GPU mode is CUDA library. The installation instruction of CUDA can be found on [http://docs.nvidia.com/cuda/cuda-installation-guide-linux/](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

Extra dependency for provided Python scripts are **python-protobuf**, **python-numpy** and **python-matplotlib**.

On Ubuntu linux, you can use command below to install all of them.
````
sudo apt-get install python-protobuf python-numpy python-matplotlib
````

### Modifiy the CMake configuration file and compile
The configuration file of CMake is located in the root directory of this program package, it's called **CMakeLists.txt**. Modify it properly according to the instruction below:

1. Change the line *set(COMPILE_MODE_CUDA true)* to *set(COMPILE_MODE_CUDA false)* if you want to compile without CUDA.
2. Uncomment line *#list(APPEND CMAKE_PREFIX_PATH "<PATH TO PROTOBUF>")* and fill the correct path to protobuf library if your protobuf is not installed in the standard path.
3. Uncomment line *#set(CUDA_TOOLKIT_ROOT_DIR <PATH TO CUDA>)* and fill the correct path to CUDA library if your CUDA is not installed in the standard path.

Compile the program using standard CMake compilation process:

1. create a work directory inside the root path of this program package. Let's call it **build**.
````
mkdir build
cd build
````
2. create Makefile from CMake configuration file.
````
cmake ..
````
3. compile the program
````
make
````

### Run test program
The commands listed below execute based on the work direcotry **build**.

1. Define the network. You can find a predefined netowk which has 10 HCUs with 10 MCUs in each HCU in *<program_root>/data/std_test_10.prototxt*. There are 3 blocks inside the network defination file: The **gen_param** which defines the simulation process, the **net_param** which defines the newtork structure and update policy and the **rec_param** which defines the logging system.
2. Prepare the stimuli. You can use the python script in *<program_root>/tools/stimgen/gen_10x10.py* to generate a stimuli file. The command is:
````
python ../tools/stimgen/gen_10x10.py ../data/stimuli_10x10.bin
````
3. Run the simulation. There are 4 parameters for the simulation program: *-n* specifies the location of network description file, *-s* specifies the snapshot file, *-m* specifies the mode (CPU or GPU) and *-l* forces the program to print logs to STDOUT. The command is:
````
./gsbn_sim -n ../data/std_test_10.prototxt -m GPU -l
````
4. Check the recorded spikes. There are several procedures which record the state of the program. *ProcSpkRec* records spikes, *ProcSnapshot* takes snapshot during cycles which enables snapshot, while *ProcCheck* automatically checks the correctness of patterns during test phase and prints out the overall memory capacity. You can use the python script to visulize the spikes:
````
python ../tools/plot/plot_spk.py ../data/snapshot_10/ProcSpkRec/spk_pop_0.csv
````

### Bugs
The program will crash while saving a very big snapshot file (such as the snapshot for 100x100 network). It's the problem of protobuf library.

