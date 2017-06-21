CC=g++
CFLAGS=-ffast-math -O3 -fopenmp
LINK=-lm
DEPS = dtlz.h util.h config.h

mcea_cpu: mcea.cpp dtlz.cpp util.cpp $(DEPS)
	$(CC) -o $@ $^ $(CFLAGS) $(LINK)

debug: mcea.cpp dtlz.cpp util.cpp $(DEPS)
	$(CC) -g -o mcea_cpu_dbg $^ $(CFLAGS) $(LINK)

clean:
	rm -f *.o *~
	rm mcea_cpu
