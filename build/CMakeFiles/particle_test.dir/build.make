# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

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
CMAKE_COMMAND = /afs/andrew.cmu.edu/usr20/jikaiz/private/software/bin/cmake

# The command to remove a file.
RM = /afs/andrew.cmu.edu/usr20/jikaiz/private/software/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/build

# Include any dependencies generated for this target.
include CMakeFiles/particle_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/particle_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/particle_test.dir/flags.make

CMakeFiles/particle_test.dir/src/main.cpp.o: CMakeFiles/particle_test.dir/flags.make
CMakeFiles/particle_test.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/particle_test.dir/src/main.cpp.o"
	/usr/lib64/ccache/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/particle_test.dir/src/main.cpp.o -c /afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/src/main.cpp

CMakeFiles/particle_test.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/particle_test.dir/src/main.cpp.i"
	/usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/src/main.cpp > CMakeFiles/particle_test.dir/src/main.cpp.i

CMakeFiles/particle_test.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/particle_test.dir/src/main.cpp.s"
	/usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/src/main.cpp -o CMakeFiles/particle_test.dir/src/main.cpp.s

CMakeFiles/particle_test.dir/external/glad/src/glad.c.o: CMakeFiles/particle_test.dir/flags.make
CMakeFiles/particle_test.dir/external/glad/src/glad.c.o: ../external/glad/src/glad.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/particle_test.dir/external/glad/src/glad.c.o"
	/usr/lib64/ccache/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/particle_test.dir/external/glad/src/glad.c.o   -c /afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/external/glad/src/glad.c

CMakeFiles/particle_test.dir/external/glad/src/glad.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/particle_test.dir/external/glad/src/glad.c.i"
	/usr/lib64/ccache/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/external/glad/src/glad.c > CMakeFiles/particle_test.dir/external/glad/src/glad.c.i

CMakeFiles/particle_test.dir/external/glad/src/glad.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/particle_test.dir/external/glad/src/glad.c.s"
	/usr/lib64/ccache/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/external/glad/src/glad.c -o CMakeFiles/particle_test.dir/external/glad/src/glad.c.s

# Object files for target particle_test
particle_test_OBJECTS = \
"CMakeFiles/particle_test.dir/src/main.cpp.o" \
"CMakeFiles/particle_test.dir/external/glad/src/glad.c.o"

# External object files for target particle_test
particle_test_EXTERNAL_OBJECTS =

CMakeFiles/particle_test.dir/cmake_device_link.o: CMakeFiles/particle_test.dir/src/main.cpp.o
CMakeFiles/particle_test.dir/cmake_device_link.o: CMakeFiles/particle_test.dir/external/glad/src/glad.c.o
CMakeFiles/particle_test.dir/cmake_device_link.o: CMakeFiles/particle_test.dir/build.make
CMakeFiles/particle_test.dir/cmake_device_link.o: libparticles.a
CMakeFiles/particle_test.dir/cmake_device_link.o: external/glfw/src/libglfw3.a
CMakeFiles/particle_test.dir/cmake_device_link.o: /usr/lib64/librt.so
CMakeFiles/particle_test.dir/cmake_device_link.o: /usr/lib64/libm.so
CMakeFiles/particle_test.dir/cmake_device_link.o: /usr/lib64/libX11.so
CMakeFiles/particle_test.dir/cmake_device_link.o: /usr/lib64/libXrandr.so
CMakeFiles/particle_test.dir/cmake_device_link.o: /usr/lib64/libXinerama.so
CMakeFiles/particle_test.dir/cmake_device_link.o: /usr/lib64/libXxf86vm.so
CMakeFiles/particle_test.dir/cmake_device_link.o: /usr/lib64/libXcursor.so
CMakeFiles/particle_test.dir/cmake_device_link.o: /usr/lib/gcc/x86_64-redhat-linux/4.8.5/libgomp.so
CMakeFiles/particle_test.dir/cmake_device_link.o: /usr/lib64/libpthread.so
CMakeFiles/particle_test.dir/cmake_device_link.o: CMakeFiles/particle_test.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/particle_test.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/particle_test.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/particle_test.dir/build: CMakeFiles/particle_test.dir/cmake_device_link.o

.PHONY : CMakeFiles/particle_test.dir/build

# Object files for target particle_test
particle_test_OBJECTS = \
"CMakeFiles/particle_test.dir/src/main.cpp.o" \
"CMakeFiles/particle_test.dir/external/glad/src/glad.c.o"

# External object files for target particle_test
particle_test_EXTERNAL_OBJECTS =

particle_test: CMakeFiles/particle_test.dir/src/main.cpp.o
particle_test: CMakeFiles/particle_test.dir/external/glad/src/glad.c.o
particle_test: CMakeFiles/particle_test.dir/build.make
particle_test: libparticles.a
particle_test: external/glfw/src/libglfw3.a
particle_test: /usr/lib64/librt.so
particle_test: /usr/lib64/libm.so
particle_test: /usr/lib64/libX11.so
particle_test: /usr/lib64/libXrandr.so
particle_test: /usr/lib64/libXinerama.so
particle_test: /usr/lib64/libXxf86vm.so
particle_test: /usr/lib64/libXcursor.so
particle_test: /usr/lib/gcc/x86_64-redhat-linux/4.8.5/libgomp.so
particle_test: /usr/lib64/libpthread.so
particle_test: CMakeFiles/particle_test.dir/cmake_device_link.o
particle_test: CMakeFiles/particle_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable particle_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/particle_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/particle_test.dir/build: particle_test

.PHONY : CMakeFiles/particle_test.dir/build

CMakeFiles/particle_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/particle_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/particle_test.dir/clean

CMakeFiles/particle_test.dir/depend:
	cd /afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial /afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial /afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/build /afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/build /afs/andrew.cmu.edu/usr20/jikaiz/private/15618/project/sph-tutorial/build/CMakeFiles/particle_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/particle_test.dir/depend

