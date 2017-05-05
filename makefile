CC=nvcc
CFLAGS=-arch=sm_52 --relocatable-device-code true
LINK=-lcurand
DEPS = dtlz.cuh error.h util.h

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

mcea: util.o dtlz.o mcea.o
	$(CC) -o $@ $^ $(CFLAGS) $(LINK)

clean:
	rm -f *.o *~
