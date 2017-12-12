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

## Configuration

If you want to use a different configuration for the algorithm,
you need to alter/add a file in `src/cpu/config/` (for the CPU variant) or `src/gpu/config` (for the CUDA variant).

These files are filled with simple parameter - value pairs (divided by a space).
If you leave a parameter out, a default value (from `src/cpu/config.h` or `src/gpu/config.cuh`) will be used.
The following parameters are available:

| Parameter | Description |
|-----------|-------------|
| STOPTYPE | Criterium that stops the generation loop. Available are: <br> **GENERATIONS**: number of generations that the alg. performs <br> **TIME**: the time in seconds, after which the algorithm shall be aborted *(!only in CPU variant for now)* |
| STOPVALUE | the value that is used to determine, when McEA stops (depends on the **STOPTYPE**) |
| POP_WIDTH | the y-size of the population grid (in individuals) <br> can be adjusted at will, scales the memory usage quadratical |
| N_RAD | the radius of the neighborhood around an individual <br> the neighborhood is square at all times |
| THREADS | the number of threads to use in OpenMP *(only CPU variant)*|
| VADS_SCALE | the scaling factor for the VADS weighting scheme |
| DTLZ_NUM | the number of the DTLZ problem to solve |
| PARAMS | the number of parameters for the optimization problen (DTLZ-n for now) |
| OUTFILE | the base of the filename where the results shall be written |

