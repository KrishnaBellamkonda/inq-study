# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

project(sharmonic)
cmake_minimum_required(VERSION 3.19)

if(NOT DEFINED CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "")
set(FETCHCONTENT_QUIET FALSE)
Include(FetchContent)

add_library(sharmonic INTERFACE)

target_include_directories(sharmonic INTERFACE include)

######################################
# CHECK FOR SH_LEGENDRE
######################################

include(CheckSourceCompiles)
check_source_compiles(CXX
"#include <cmath>
int main() {
    std::sph_legendre(0u, 0u, 0.0);
}
"
HAVE_SPH_LEGENDRE)

if(HAVE_SPH_LEGENDRE)
  message(STATUS "Looking for std::sph_legendre - found")
else()
  message(STATUS "Looking for std::sph_legendre - not found")
  message(WARNING "The function std::sph_legendre was not found. Thorough testing will be disabled.")  
endif()

######################################
# CHECK FOR BOOST MATH
######################################

if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
  find_package(Boost COMPONENTS math)
else()
  find_package(Boost)
endif()

if(Boost_FOUND)
  message(STATUS "Boost actual version found " ${Boost_VERSION})
  if(Boost_VERSION LESS_EQUAL "1.53.0")  # 105300
    message(STATUS "Applying option BOOST_NO_AUTO_PTR to patch for old Boost")
	  add_compile_options(-DBOOST_NO_AUTO_PTR)
  endif()

  if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
    target_link_libraries(sharmonic INTERFACE Boost::math)
  else()
    target_include_directories(sharmonic INTERFACE Boost::boost)
  endif()

  target_compile_definitions(sharmonic INTERFACE "BOOST_PP_VARIADICS=1")  # NVCC needs BOOST_PP_VARIADICS to compile some Boost headers
else()
  message(WARNING "Cannot found boost. Thorough testing will be disabled.")
endif()

######################################
# CHECK FOR GSL
######################################

find_package(GSL)

if(GSL_FOUND)
  target_link_libraries(sharmonic INTERFACE GSL::gsl)
endif()

###########################
# GPU SUPPORT (CUDA & HIP)
###########################

if(ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 11.5 REQUIRED)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    SET(CMAKE_CUDA_ARCHITECTURES native)
  endif()

	set(ENABLE_GPU TRUE)
	set(GPU_LANGUAGE CUDA)
  set(CMAKE_CUDA_FLAGS  "--expt-relaxed-constexpr ${CMAKE_CUDA_FLAGS}")
endif()

if(ENABLE_HIP)
  enable_language(HIP)

	find_package(hip REQUIRED)

	set(ENABLE_GPU TRUE)
	set(GPU_LANGUAGE HIP)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

########################
# ALWAYS OPTIMIZE THE CODE
########################

if(ENABLE_CUDA)
  set(CMAKE_CXX_FLAGS  "-O3 -ffast-math ${CMAKE_CXX_FLAGS} -fno-lto")  # in newer versions of Ubuntu, LTO is on by default, which is incompatible with NVCC
else()
  set(CMAKE_CXX_FLAGS  "-O3 -ffast-math ${CMAKE_CXX_FLAGS}")
endif()

########################
# CATCH2
########################

message(STATUS "Downloading Catch2 v3p3p2")
FetchContent_Declare(
	Catch2
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG        v3.3.2
  GIT_PROGRESS TRUE
	)
FetchContent_MakeAvailable(Catch2)
include_directories(${catch2_SOURCE_DIR}/src)

###############################
# GENERATE CONFIGURE FILE
###############################

configure_file(sharmonic_config.h.in sharmonic_config.h)
target_compile_definitions(sharmonic INTERFACE "HAVE_SHARMONIC_CONFIG_H=1")

############################
# UNIT TESTS
############################

include(CTest)

include_directories(${PROJECT_BINARY_DIR})

add_executable(reference)
set_property(TARGET reference PROPERTY CXX_STANDARD 17)
target_sources(reference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/reference.cpp)
if(ENABLE_GPU)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/tests/reference.cpp PROPERTIES LANGUAGE ${GPU_LANGUAGE})
endif()
target_link_libraries(reference PRIVATE Catch2::Catch2WithMain sharmonic)
add_test(NAME reference COMMAND ./reference)

add_executable(gradients)
set_property(TARGET gradients PROPERTY CXX_STANDARD 17)
target_sources(gradients PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/gradients.cpp)
if(ENABLE_GPU)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/tests/gradients.cpp PROPERTIES LANGUAGE ${GPU_LANGUAGE})
endif()
target_link_libraries(gradients PRIVATE Catch2::Catch2WithMain sharmonic)
add_test(NAME gradients COMMAND ./gradients)

add_executable(hessian)
set_property(TARGET hessian PROPERTY CXX_STANDARD 17)
target_sources(hessian PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/hessian.cpp)
if(ENABLE_GPU)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/tests/hessian.cpp PROPERTIES LANGUAGE ${GPU_LANGUAGE})
endif()
target_link_libraries(hessian PRIVATE Catch2::Catch2WithMain sharmonic)
add_test(NAME hessian COMMAND ./hessian)

add_executable(legendre)
set_property(TARGET legendre PROPERTY CXX_STANDARD 17)
target_sources(legendre PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/legendre.cpp)
if(ENABLE_GPU)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/tests/legendre.cpp PROPERTIES LANGUAGE ${GPU_LANGUAGE})
endif()
target_link_libraries(legendre PRIVATE Catch2::Catch2WithMain sharmonic)
add_test(NAME legendre COMMAND ./legendre)

add_executable(internal)
set_property(TARGET internal PROPERTY CXX_STANDARD 17)
target_sources(internal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/internal.cpp)
if(ENABLE_GPU)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/tests/internal.cpp PROPERTIES LANGUAGE ${GPU_LANGUAGE})
endif()
target_link_libraries(internal PRIVATE Catch2::Catch2WithMain sharmonic)
add_test(NAME internal COMMAND ./internal)

add_executable(c_test)
set_property(TARGET c_test PROPERTY C_STANDARD 99)
target_sources(c_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/c_test.c)
target_link_libraries(c_test PRIVATE sharmonic m)
add_test(NAME c_test COMMAND ./c_test)

add_executable(benchmark)
set_property(TARGET benchmark PROPERTY CXX_STANDARD 17)
target_sources(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark.cpp)
if(ENABLE_GPU)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark.cpp PROPERTIES LANGUAGE ${GPU_LANGUAGE})
endif()
target_link_libraries(benchmark PRIVATE sharmonic)

