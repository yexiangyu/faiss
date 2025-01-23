#!/bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e


# Build libfaiss.so/libfaiss_avx2.so/libfaiss_avx512.so
cmake -B _build \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTING=OFF \
      -DFAISS_OPT_LEVEL=avx512 \
      -DFAISS_ENABLE_GPU=OFF \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DBLA_VENDOR=Intel10_64lp \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=Debug .

make -C _build -j$(nproc) faiss faiss_avx2 faiss_avx512

cmake --install _build --prefix $PREFIX
cmake --install _build --prefix _libfaiss_stage/
