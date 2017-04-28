CC=nvcc
CFLAGS=-arch=sm_52 --relocatable-device-code true 
DEPS = dtlz.cuh error.h util.h

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

mcea: util.o dtlz.o mcea.o
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o *~
