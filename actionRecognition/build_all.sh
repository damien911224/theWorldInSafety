#!/usr/bin/env bash

version="2.4.13"

echo "Building OpenCV" $version
cd 3rd-party/opencv-$version
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_TBB=ON  -D WITH_V4L=ON ..
if make; then
    cp lib/cv2.so ../../../
    echo "OpenCV" $version "built."
else
    echo "Failed to build OpenCV. Please check the logs above."
    exit 1
fi

cd ../../..

sudo apt-get -qq install libzip-dev -y

cd lib/dense_flow_cpu
cd build
OpenCV_DIR=../../../3rd-party/opencv-2.4.13/build/ cmake .. -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
if make; then
    echo "Dense Flow built."
else
    echo "Failed to build Dense Flow. Please check the logs above."
    exit 1
fi

# build caffe
# echo "Building Caffe, MPI status: ${CAFFE_USE_MPI}"
# cd lib/caffe-action
# cd build
# if [ "$CAFFE_USE_MPI" == "MPI_ON" ]; then
# OpenCV_DIR=../../../3rd-party/opencv-$version/build/ cmake .. -DUSE_MPI=ON -DMPI_CXX_COMPILER="${CAFFE_MPI_PREFIX}/bin/mpicxx" -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
# else
# OpenCV_DIR=../../../3rd-party/opencv-$version/build/ cmake .. -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
# fi
# if make -j8; then
#     echo "Caffe Built."
#     echo "All tools built. Happy experimenting!"
#     cd ../../../
# else
#     echo "Failed to build Caffe. Please check the logs above."
#     exit 1
# fi
