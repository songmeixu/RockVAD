#!/usr/bin/env bash

# export PATH=/root/toolchain-aarch64_generic-glibc-1130/bin:$PATH
# export PATH=/audio/work/npu/mr527/docker/docker_data/toolchain-aarch64_generic-glibc-1130/bin:$PATH
if ! command -v aarch64-openwrt-linux-gcc &> /dev/null; then
  echo "Please set a toolchain for cross-compiling."
  exit 1
fi

set -ex

dir=build-aarch64-linux-mr527-release
mkdir -p $dir
cd $dir

# if [ ! -f alsa-lib/src/.libs/libasound.so ]; then
#   echo "Start to cross-compile alsa-lib"
#   if [ ! -d alsa-lib ]; then
#     git clone --depth 1 https://github.com/alsa-project/alsa-lib
#   fi
#   # If it shows:
#   #  ./gitcompile: line 79: libtoolize: command not found
#   # Please use:
#   #  sudo apt-get install libtool m4 automake
#   #
#   pushd alsa-lib
#   CC=aarch64-openwrt-linux-gcc ./gitcompile --host=aarch64-linux-gnu
#   popd
#   echo "Finish cross-compiling alsa-lib"
# fi

# export CPLUS_INCLUDE_PATH=$PWD/alsa-lib/include:$CPLUS_INCLUDE_PATH
# export ROCKVAD_ALSA_LIB_DIR=$PWD/alsa-lib/src/.libs

cmake \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DROCKVAD_ENABLE_PYTHON=OFF \
  -DROCKVAD_ENABLE_PORTAUDIO=OFF \
  -DROCKVAD_ENABLE_JNI=OFF \
  -DROCKVAD_ENABLE_BINARY=ON \
  -DROCKVAD_ENABLE_CPP_API=ON \
  -DROCKVAD_ENABLE_TEST=OFF \
  -DROCKVAD_ENABLE_NPU=OFF \
  -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-mr527.toolchain.cmake \
  ..

make VERBOSE=1 -j1
make install/strip

# cp -v $ROCKVAD_ALSA_LIB_DIR/libasound.so* ./install/lib/
