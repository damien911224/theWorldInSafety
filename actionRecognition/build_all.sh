#!/usr/bin/env bash
CAFFE_USE_MPI=${1:-OFF}
CAFFE_MPI_PREFIX=${MPI_PREFIX:-""}

version="2.4.13"

cd lib/dense_flow
cd build
OpenCV_DIR=../../../3rd-party/opencv-2.4.13/build/ cmake .. -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
if make -j8 ; then
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
