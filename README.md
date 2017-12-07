# McEA
Multicriterial cellular Evolutionary Algorithm

## Usage

There is a GPU variant of the algorithm residing in `src/gpu/` and the CPU variant in `src/cpu/`.
Test them via:

```
mkdir -p build/release
cd build/release
cmake -DCMAKE_BUILD_TYPE=Release -DMCEA_CUDA_ARCH=<insert arch number here (e.g.: 52)> ../..
make
```

There is also a `-DCMAKE_BUILD_TYPE=Debug` build type available and if you omit the `-DMCEA_CUDA_ARCH` cmake will try to find the cuda architecture itself (linux only).
If you use `ccmake`, you can also set the `MCEA_CUDA_ARCH` permanently to a cached value.


After that multiple variants with different parameters are located in `build/release/gpu/` or `build.release/cpu/`.
The used parameters are coded into the binary-name like this:

```
  mcea_<platform>_g<generations>_pw<population width>_p<dtlz parameters>_r<neighbourhood radius>_t<threads on cpu>_vs<vads scaling factor>
```

You need a NVidia GPU and the CUDA framework for this obviously.
Also you need cmake (min 3.0.2) and make for compilation.
All the DTLZ problems (1-7) can be optimized, just change the `DTLZ_NUM` parameter in the config files located in `src/cpu|gpu/config/`.
Also a documentation how the algorithm works is coming.

To create a doc from the source run `build_docs.sh` in `doc/`.
