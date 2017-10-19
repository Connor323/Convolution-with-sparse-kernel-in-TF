cmake_minimum_required(VERSION 3.5)

execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)

include_directories(${Tensorflow_INCLUDE_DIRS})

find_package(CUDA)

set (CMAKE_CXX_FLAGS "--std=c++11 -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1")

set(CMAKE_BUILD_TYPE Debug)

#pass flags to c++ compiler
SET(CUDA_PROPAGATE_HOST_FLAGS ON)

#create library
cuda_add_library(
	conv_sparse SHARED
	conv_sparse.cu
	conv_sparse.cc)

#copy python files to build folder (for easy testing)
file(GLOB PY_FILES
  "src/*.py"
)
file(COPY ${PY_FILES} DESTINATION .)
