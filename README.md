# GPGPU Motion Detection with CUDA and GStreamer

This project implements a **GStreamer plugin** for **real-time motion detection**.  
It provides two implementations:
- A **CPU version** in C++
- A **GPU-accelerated version** in CUDA  

The plugin processes video streams and highlights detected motion in **red**, enabling high-performance video analysis.

---

## Features
- Real-time motion detection on video streams  
- CUDA-accelerated processing for improved FPS  
- Morphological filtering and hysteresis thresholding for robust detection  
- Compatible with **GStreamer pipelines**  
- Works with both **video files** and **live webcam input**  

---

## Installation

### 1. Prerequisites
- **GStreamer 1.0** (and development headers)  
- **CUDA Toolkit** (if using GPU acceleration)  
- CMake (≥ 3.10)  

On Ubuntu:
```bash
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
```

## Build the Plugin
```bash
cmake -S . -B build --preset release -D USE_CUDA=ON   # or OFF for CPU version
cmake --build build
```

## Environment Setup
```bash
# Download example video
wget https://cloud.lrde.epita.fr/s/tyeqDFYfXM8i3km/download -O video03.avi

# Export plugin path
export GST_PLUGIN_PATH=$(pwd)

# Link plugin (choose CPU or CUDA version)
ln -s ./build/libgstcudafilter-cpp.so libgstcudafilter.so   # CPU
ln -s ./build/libgstcudafilter-cuda.so libgstcudafilter.so  # CUDA
```

---

## Usage
### Process a Local Video and Save to MP4
```bash
gst-launch-1.0 uridecodebin uri=file://$(pwd)/video03.avi ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=output.mp4
```

### Live Webcam Stream with FPS Display
```bash
gst-launch-1.0 -e -v v4l2src ! jpegdec ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink
```

### Local Video Playback with FPS Display
```bash
gst-launch-1.0 -e -v uridecodebin uri=file://$(pwd)/video03.avi ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink
```

### Benchmarking FPS (no display)
```bash
gst-launch-1.0 -e -v uridecodebin uri=file://$(pwd)/video03.avi ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink video-sink=fakesink sync=false
```

---

## Benchmarking with Nsight Systems
1. Generate reports with nsys
2. Convert reports to SQLite
3. Move `.sqlite` files into `reports/`
4. Extract data for plots:
```bash
python extract_plot_data.py ../reports/vX/ --version "Version X"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 plot_data.ipynb
python extract_latex_data.py ../reports/vX/ --version "Version X"
python generate_latex_tables.py
```

---

## Authors
- Thomas Galateau
- Nathan Sue
- Jules-Victor Lépinay
