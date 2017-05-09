CC=nvcc
CFLAGS=-arch=sm_52 --relocatable-device-code true -lineinfo
LINK=-lcurand
DEPS = dtlz.cuh error.h util.h

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

mcea: util.o dtlz.o mcea.o
	$(CC) -o $@ $^ $(CFLAGS) $(LINK)

debug: mcea.cu dtlz.cu util.cu
	$(CC) -g -G -o mcea_dbg $^ $(CFLAGS) $(LINK)

clean:
	rm -f *.o *~
