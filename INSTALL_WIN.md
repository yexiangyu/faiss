# Build faiss & faiss_c on windows (no python) 


## Install Dependencies
- Intel MKL
- CUDA
- CMake

## Build

```shell
# config cmake
cmake -B build -Ax64 -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF

# build 
cmake --build build --config Release

# install
cmake --install build --prefix c:\tools\faiss
```


