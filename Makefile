# Makefile for fft_convolution (cuFFT + OpenCV)
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

SRC := src/fft_convolution.cu
BIN := bin/fft_filter

NVCC_FLAGS := -std=c++17 -O3 $(OPENCV_CFLAGS) -I$(CUDA_PATH)/include
LD_FLAGS := -L$(CUDA_PATH)/lib64 -lcufft -lcudart $(OPENCV_LIBS)

.PHONY: all build run clean

all: build

build: $(BIN)

$(BIN): $(SRC)
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LD_FLAGS) -Wno-deprecated-gpu-targets
	@echo "Built $@"

run: $(BIN)
	@mkdir -p output
	./$(BIN) data/input.png output/output.png

clean:
	rm -rf bin output
