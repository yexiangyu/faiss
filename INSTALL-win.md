# Building Faiss on Windows

This document describes how to build Faiss with CUDA GPU support and Intel MKL on Windows.

## Prerequisites

- **Operating System**: Windows 10/11 64-bit
- **IDE**: Visual Studio 2022 (Community or higher)
- **CMake**: 3.17 or later
- **CUDA Toolkit**: 12.x or 13.x (tested with CUDA 13.1)
- **Intel oneAPI MKL**: 2024.x or later (tested with MKL 2025.3)
- **Git**: for cloning the repository

### 1. Install Visual Studio 2022

Download and install Visual Studio 2022 Community from:
https://visualstudio.microsoft.com/downloads/

Select the following workloads:
- **Desktop development with C++**
- **CUDA development** (if available, otherwise install CUDA separately)

### 2. Install CUDA Toolkit

Download CUDA Toolkit from:
https://developer.nvidia.com/cuda-downloads

Recommended version: CUDA 13.1

After installation, verify:
```
nvcc --version
```

### 3. Install Intel oneAPI MKL

Download Intel oneAPI Base Toolkit from:
https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

Or install standalone MKL:
https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

Default installation path: `C:\Program Files (x86)\Intel\oneAPI\mkl\2025.3`

### 4. Clone Faiss Repository

```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss
git checkout v1.14.1  # or desired version
git checkout -b windows-build
```

## Build Configuration

### CMake Configure

```bash
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
  -DFAISS_ENABLE_GPU=ON ^
  -DFAISS_ENABLE_PYTHON=OFF ^
  -DBUILD_TESTING=OFF ^
  -DBUILD_SHARED_LIBS=ON ^
  -DFAISS_ENABLE_C_API=ON ^
  -DMKL_ROOT="C:/Program Files (x86)/Intel/oneAPI/mkl/2025.3"
```

### CMake Options

| Option | Description |
|--------|-------------|
| `FAISS_ENABLE_GPU` | Enable CUDA GPU support |
| `FAISS_ENABLE_PYTHON` | Build Python bindings (requires Python) |
| `BUILD_TESTING` | Build test executables |
| `BUILD_SHARED_LIBS` | Build shared libraries (DLL) |
| `FAISS_ENABLE_C_API` | Build C API |
| `MKL_ROOT` | Path to Intel MKL installation |

## Build

### Build All Targets

```bash
cmake --build build --config Release
```

### Build Specific Target

```bash
# Build only faiss library
cmake --build build --config Release --target faiss

# Build only C API
cmake --build build --config Release --target faiss_c
```

## Output Files

After successful build, the following files are generated:

| File | Description |
|------|-------------|
| `build/faiss/Release/faiss.dll` | Main Faiss library |
| `build/faiss/Release/faiss.lib` | Import library for faiss.dll |
| `build/c_api/Release/faiss_c.dll` | C API library |
| `build/c_api/Release/faiss_c.lib` | Import library for faiss_c.dll |

## Installation

```bash
cmake --install build --prefix "C:/path/to/install"
```

This installs:
- Libraries to `<prefix>/lib/`
- Headers to `<prefix>/include/faiss/`
- C API headers to `<prefix>/include/faiss/c_api/`

## Code Changes Required for Windows

The following changes are required to compile Faiss on Windows with CUDA:

### 1. CMakeLists.txt

Add compiler flags for UTF-8 encoding and define NOMINMAX:

```cmake
if(MSVC)
  add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
  add_compile_definitions(NOMINMAX)
endif()
```

### 2. CUDA Template Function Fix

The CUDA 13.1 compiler cannot properly deduce `std::initializer_list` in template function calls like `tensor.view<N>({a, b})`. 

**File: `faiss/gpu/utils/Tensor.cuh`**

Add array parameter overload:

```cpp
template <int NewDim>
__host__ __device__ Tensor<T, NewDim, InnerContig, IndexT, PtrTraits> view(
        const IndexT (&sizes)[NewDim]);
```

**File: `faiss/gpu/utils/Tensor-inl.cuh`**

Add implementation:

```cpp
template <typename T, int Dim, bool InnerContig, typename IndexT, template <typename U> class PtrTraits>
template <int NewDim>
__host__ __device__ Tensor<T, NewDim, InnerContig, IndexT, PtrTraits> Tensor<
        T,
        Dim,
        InnerContig,
        IndexT,
        PtrTraits>::view(const IndexT (&sizes)[NewDim]) {
    GPU_FAISS_ASSERT(this->isContiguous());

    size_t curSize = numElements();
    size_t newSize = 1;

    for (int i = 0; i < NewDim; ++i) {
        newSize *= sizes[i];
    }

    GPU_FAISS_ASSERT(curSize == newSize);
    return Tensor<T, NewDim, true, IndexT, PtrTraits>(data(), sizes);
}
```

### 3. Windows SDK Macro Conflict

Windows SDK defines `small` as a macro in `rpcndr.h`, which conflicts with variable names.

**File: `faiss/gpu/utils/MergeNetworkWarp.cuh`**

Rename `small` variable to `isSmaller`:

```cpp
// Before
bool small = ...;

// After
bool isSmaller = ...;
```

### 4. Update view() Calls

Replace braced initializer list calls with array syntax:

```cpp
// Before
auto view = tensor.view<3>({a, b, c});

// After
idx_t sizes[3] = {a, b, c};
auto view = tensor.view<3>(sizes);
```

Affected files:
- `faiss/gpu/impl/PQCodeDistances-inl.cuh`
- `faiss/gpu/impl/IVFPQ.cu`

### 5. C API DLL Export

Add DLL export macros for Windows.

**File: `c_api/faiss_c.h`**

```cpp
#ifdef _WIN32

#ifdef FAISS_C_MAIN_LIB
#define FAISS_C_API __declspec(dllexport)
#else
#define FAISS_C_API __declspec(dllimport)
#endif

#else

#define FAISS_C_API

#endif
```

Update all function declaration macros:

```cpp
#define FAISS_DECLARE_GETTER(clazz, ty, name) \
    FAISS_C_API ty faiss_##clazz##_##name(const Faiss##clazz*);

#define FAISS_DECLARE_SETTER(clazz, ty, name) \
    FAISS_C_API void faiss_##clazz##_set_##name(Faiss##clazz*, ty);

#define FAISS_DECLARE_DESTRUCTOR(clazz) \
    FAISS_C_API void faiss_##clazz##_free(Faiss##clazz* obj);
```

**File: `c_api/CMakeLists.txt`**

```cmake
target_compile_definitions(faiss_c PRIVATE FAISS_C_MAIN_LIB)
```

## Known Issues

1. **C4819 Warnings**: These are encoding warnings from CUDA headers and can be safely ignored.

2. **CUDA Compilation Time**: CUDA files take a long time to compile. Be patient.

3. **MSVC Traditional Preprocessor**: Some CUDA code may trigger warnings about using the traditional preprocessor. These are informational only.

## Usage Example

### Linking with faiss_c.dll

```c
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/IndexFlat_c.h>

int main() {
    FaissIndexFlat* index;
    int d = 128;
    
    if (faiss_IndexFlatL2_new_with(&index, d) != 0) {
        // Handle error
        return -1;
    }
    
    // Add vectors
    float* x = ...; // your vectors
    faiss_Index_add((FaissIndex*)index, n, x);
    
    // Search
    float* distances = malloc(k * n * sizeof(float));
    idx_t* labels = malloc(k * n * sizeof(idx_t));
    faiss_Index_search((FaissIndex*)index, n, x, k, distances, labels);
    
    faiss_IndexFlat_free(index);
    return 0;
}
```

### Linking with faiss.dll (C++)

```cpp
#include <faiss/IndexFlat.h>

int main() {
    int d = 128;
    faiss::IndexFlatL2 index(d);
    
    // Add vectors
    index.add(n, x);
    
    // Search
    std::vector<float> distances(k * n);
    std::vector<faiss::idx_t> labels(k * n);
    index.search(n, x, k, distances.data(), labels.data());
    
    return 0;
}
```

## Troubleshooting

### CUDA not found

Make sure CUDA_PATH environment variable is set:
```
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
```

### MKL not found

Specify MKL_ROOT explicitly in CMake command:
```
-DMKL_ROOT="C:/Program Files (x86)/Intel/oneAPI/mkl/2025.3"
```

### Linker errors

Make sure all required DLLs are in PATH or copied to the executable directory:
- faiss.dll
- faiss_c.dll
- CUDA DLLs (cudart64_*.dll, cublas64_*.dll, cublasLt64_*.dll)
- MKL DLLs (mkl_core.2.dll, mkl_intel_thread.2.dll, mkl_rt.2.dll)

## License

Faiss is MIT licensed. See LICENSE file for details.