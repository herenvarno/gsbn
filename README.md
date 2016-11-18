# GSBN
 
GPU version of Spiking-based BCPNN.

## Compilation

GSBN can be compiled with or without CUDA. If CUDA is not present, the program can only run in CPU_ONLY mode.

### Dependencies
To compile the full version of GSBN, Protobuf and CUDA are needed. Use command below to install protobuf:

#### Debian/Ubuntu
````
sudo apt-get install protobuf
````
#### Fedora/CentOS
````
sudo yum install protobuf
````
#### Archlinux
````
sudo pacman -S protobuf
````

The installation instruction of CUDA can be found on [http://docs.nvidia.com/cuda/cuda-installation-guide-linux/](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

If you want to use the provided python scripts, addition libraries for python are needed. The libraries are **protobuf-python**, **numpy** and **matplotlib**.

### Compilation
#### Modify the CMake configuration file
The configuration file of CMake is located in the root directory of this program package, it's called **CMakeLists.txt**.

1. Change the line *set(COMPILE_MODE_CUDA true)* to *set(COMPILE_MODE_CUDA false)* if you want to compile without CUDA.
2. Uncomment line *#list(APPEND CMAKE_PREFIX_PATH "<PATH TO PROTOBUF>")* and fill the correct path to protobuf library if your protobuf is not installed in the standard path.
3. Uncomment line *#set(CUDA_TOOLKIT_ROOT_DIR <PATH TO CUDA>)* and fill the correct path to CUDA library if your CUDA is not installed in the standard path.

#### Compile
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
4. Check the recorded spikes. If you set the **rec_param** to allow spike recording, a file called *spike.csv* will be created in output directory. You can use the python script to visulize the spikes:
````
python ../tools/plot/plot_spk.py ../data/snapshot_10/spike.csv 0
````

### Bugs
The program will crash while saving a very big snapshot file (such as the snapshot for 100x100 network). It's the problem of protobuf library.

CMake with version less than 3.4 doesn't support PROTOBUF_GENERATE_PYTHON command. If you want to use the provided python script for debugging. You need to generate the python package for *<program root>/src/proto/gsbn.proto* to **build** directory manually. Or just copy the file *<program root>/proto_python/gsbn_pb2.py* to **build** directory.
 

