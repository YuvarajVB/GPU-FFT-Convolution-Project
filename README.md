# GPU FFT Convolution Project

## Overview
Performs 2D image convolution on a GPU using cuFFT + CUDA.
- Input: `data/input.png` (grayscale)
- Kernel: 7×7 box blur
- Output: `output/output.png`

## Steps to Build and Run

1.  Put your input PNG in `data/input.png`\

2.  Build the project:

    ``` bash
    make build
    ```

3.  Run the program:

    ``` bash
    ./bin/fft_filter data/input.png output/output.png
    ```

    Or use:

    ``` bash
    chmod +x run.sh
    ./run.sh
    ```

##  How It Works 

-   Reads and pads the input image to the next power-of-two size\
-   Uses cuFFT to compute the FFT of the image and kernel\
-   Multiplies them in the frequency domain\
-   Inverse FFT converts back to spatial domain\
-   Crops and saves PNG output
