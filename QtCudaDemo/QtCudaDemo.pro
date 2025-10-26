QT += core gui widgets

CONFIG += c++17
CONFIG -= app_bundle

SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui

# --- CUDA setup ---
CUDA_DIR = /usr/local/cuda
INCLUDEPATH += $$CUDA_DIR/include
LIBS += -L$$CUDA_DIR/lib64 -lcudart

CUDA_SOURCES = cuda/kernels.cu
CUDA_OBJECTS_DIR = $$OUT_PWD/cuda

# Define CUDA compiler
NVCC = $$CUDA_DIR/bin/nvcc

# Custom rule to compile .cu files
cuda.input = CUDA_SOURCES
cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
cuda.commands = $$NVCC -c -std=c++17 -Xcompiler -fPIC \
                -I$$INCLUDEPATH \
                $$CUDA_SOURCES -o $$cuda.output
cuda.dependency_type = TYPE_C
cuda.CONFIG += no_link

QMAKE_EXTRA_COMPILERS += cuda

# Link the compiled CUDA object file with the main app
OBJECTS += $$CUDA_OBJECTS_DIR/kernels_cuda.o

# Deployment
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
