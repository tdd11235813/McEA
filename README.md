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
For now there is only source code doc in `gpu/doc/`.

## CPU

For comparison reasons a OpenACC CPU variant is coming.
