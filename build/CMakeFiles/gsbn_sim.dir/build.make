# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/herenvarno/Project/gsbn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/herenvarno/Project/gsbn/build

# Include any dependencies generated for this target.
include CMakeFiles/gsbn_sim.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/gsbn_sim.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gsbn_sim.dir/flags.make

CMakeFiles/gsbn_sim.dir/src/gsbn_sim/Main.cpp.o: CMakeFiles/gsbn_sim.dir/flags.make
CMakeFiles/gsbn_sim.dir/src/gsbn_sim/Main.cpp.o: ../src/gsbn_sim/Main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/herenvarno/Project/gsbn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/gsbn_sim.dir/src/gsbn_sim/Main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gsbn_sim.dir/src/gsbn_sim/Main.cpp.o -c /home/herenvarno/Project/gsbn/src/gsbn_sim/Main.cpp

CMakeFiles/gsbn_sim.dir/src/gsbn_sim/Main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gsbn_sim.dir/src/gsbn_sim/Main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/herenvarno/Project/gsbn/src/gsbn_sim/Main.cpp > CMakeFiles/gsbn_sim.dir/src/gsbn_sim/Main.cpp.i

CMakeFiles/gsbn_sim.dir/src/gsbn_sim/Main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gsbn_sim.dir/src/gsbn_sim/Main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/herenvarno/Project/gsbn/src/gsbn_sim/Main.cpp -o CMakeFiles/gsbn_sim.dir/src/gsbn_sim/Main.cpp.s

# Object files for target gsbn_sim
gsbn_sim_OBJECTS = \
"CMakeFiles/gsbn_sim.dir/src/gsbn_sim/Main.cpp.o"

# External object files for target gsbn_sim
gsbn_sim_EXTERNAL_OBJECTS =

gsbn_sim: CMakeFiles/gsbn_sim.dir/src/gsbn_sim/Main.cpp.o
gsbn_sim: CMakeFiles/gsbn_sim.dir/build.make
gsbn_sim: libgsbn.so
gsbn_sim: /usr/lib64/libprotobuf.so
gsbn_sim: /usr/lib64/openmpi/lib/libmpi_cxx.so
gsbn_sim: /usr/lib64/openmpi/lib/libmpi.so
gsbn_sim: CMakeFiles/gsbn_sim.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/herenvarno/Project/gsbn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable gsbn_sim"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gsbn_sim.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gsbn_sim.dir/build: gsbn_sim

.PHONY : CMakeFiles/gsbn_sim.dir/build

CMakeFiles/gsbn_sim.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gsbn_sim.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gsbn_sim.dir/clean

CMakeFiles/gsbn_sim.dir/depend:
	cd /home/herenvarno/Project/gsbn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/herenvarno/Project/gsbn /home/herenvarno/Project/gsbn /home/herenvarno/Project/gsbn/build /home/herenvarno/Project/gsbn/build /home/herenvarno/Project/gsbn/build/CMakeFiles/gsbn_sim.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gsbn_sim.dir/depend

