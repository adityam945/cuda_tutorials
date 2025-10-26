TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle

TARGET = cuda_console_demo

SOURCES += main.cpp

# --- 1. CUDA Environment Setup (Confirmed Path) ---
CUDA_DIR = /usr/local/cuda-13.0
INCLUDEPATH += $$CUDA_DIR/include

# Link the CUDA Runtime Library
LIBS += -L$$CUDA_DIR/lib64 -lcudart
LIBS += -lstdc++

# --- 2. Manual CUDA Compilation (Bypassing QMAKE_EXTRA_COMPILERS) ---
# Define paths
NVCC = $$CUDA_DIR/bin/nvcc
CUDA_SOURCE = $$PWD/cuda/kernels.cu
CUDA_OBJECT_DIR = $$OUT_PWD/cuda
CUDA_OBJECT = $$CUDA_OBJECT_DIR/kernels.o

# Ensure the object directory exists
!exists($$CUDA_OBJECT_DIR): system(mkdir -p $$CUDA_OBJECT_DIR)

# Tell QMake to include the pre-compiled CUDA object file in the link step
OBJECTS += $$CUDA_OBJECT

# Define the actual build command as a PRE_TARGETDEPS (dependency)
# The $$quote() ensures the paths with spaces or special characters are handled correctly.
# NOTE: We force the output to the correct location for the linker.
CUDA_COMMAND = $$quote($$NVCC) -c -std=c++17 -Xcompiler -fPIC \
    -I$$quote($$CUDA_DIR/include) $$quote($$CUDA_SOURCE) -o $$quote($$CUDA_OBJECT)

# Add the command as a custom step dependency to run before linking the target
# This creates a dummy target in the Makefile to run NVCC.
QMAKE_EXTRA_TARGETS += cuda_compile_step
cuda_compile_step.target = $$CUDA_OBJECT
cuda_compile_step.depends = $$CUDA_SOURCE
cuda_compile_step.commands = $$CUDA_COMMAND
PRE_TARGETDEPS += $$CUDA_OBJECT

# Remove the previously problematic custom compiler section
QMAKE_EXTRA_COMPILERS -= cuda