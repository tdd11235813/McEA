# McEA
Multicriterial cellular Evolutionary Algorithm

## Usage

There is a GPU variant of the algorithm residing in `gpu/` and the CPU variant in `cpu/`.
Test them via:

```
cmake .
make
```

... in the respetive directories.

After that multiple variants with different parameters are located in `/gpu/bin/` or `cpu/bin`.
The used parameters are coded into the binary-name like this:

```
  mcea_<platform>_g<generations>_pw<population width>_p<dtlz parameters>_r<neighbourhood radius>_t<threads on cpu>_vs<vads scaling factor>
```

You need a NVidia GPU and the CUDA framework for this obviously.
Also you need cmake and make for compilation.
All the DTLZ problems (1-7) can be optimized, but the mcea.<cu|cpp> has to be altered.
Just change all occurrences of the dtlzX() function (X is the problem number) to your preferred variant.
Also a documentation how the algorithm works is coming.

To create a doc from the source run `doxygen` in `gpu/` and `cpu/`.
