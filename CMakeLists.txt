cmake_minimum_required(VERSION 2.8.9)
project(gsbn)
# The project version number.
set(VERSION_MAJOR 0 CACHE STRING "Project major version number.")
set(VERSION_MINOR 0 CACHE STRING "Project minor version number.")
set(VERSION_PATCH 1 CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

include_directories(include)
file(GLOB SOURCES "src/*/*.cpp")
set(CMAKE_CXX_FLAGS "-std=c++11 -w -pedantic -g -O2")
set(PROJECT_LINK_LIBS )

add_executable(gsbn ${SOURCES})
target_link_libraries(gsbn ${PROJECT_LINK_LIBS})

#add_executable(test_Mcu test/ghcu/Mcu.cpp)
#target_link_libraries(test_Mcu ${PROJECT_LINK_LIBS})

enable_testing()
#add_test(test_Mcu test_Mcu)

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" ${DOXYGEN_FOUND})

if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
endif()
