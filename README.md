# McEA
Multicriterial cellular Evolutionary Algorithm

## GPU

There is a GPU variant of the algorithm residing in `gpu/`.
Test it via:

```
make
./mcea
```

You need a NVidia GPU and the CUDA framework for this obviously.
For now only the DTLZ1 problem is optimized. More to follow.
Also a documentation how the algorithm works is coming.

To create a doc from the source run `doxygen` in `gpu/`.

## CPU

There is a GPU variant of the algorithm residing in `cpu/`.
Test it via:

```
make
./mcea_cpu
```

You need a g++ compiler version with OpenMP support for that
The same optimization problems as in the GPU varant are available.

To create a doc from the source run `doxygen` in `cpu/`.
