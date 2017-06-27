CC=g++
CFLAGS=-ffast-math -O3 -fopenmp
LINK=-lm
DEPS = dtlz.h util.h config.h

all: mcea_cpu_g10 mcea_cpu_g100 mcea_cpu_g1000

mcea_cpu_g10: mcea.cpp dtlz.cpp util.cpp $(DEPS)
	$(CC) -o bin/$@ $^ $(CFLAGS) $(LINK) -DGENERATIONS=10

mcea_cpu_g100: mcea.cpp dtlz.cpp util.cpp $(DEPS)
	$(CC) -o bin/$@ $^ $(CFLAGS) $(LINK) -DGENERATIONS=100

mcea_cpu_g1000: mcea.cpp dtlz.cpp util.cpp $(DEPS)
	$(CC) -o bin/$@ $^ $(CFLAGS) $(LINK) -DGENERATIONS=1000

debug: mcea.cpp dtlz.cpp util.cpp $(DEPS)
	$(CC) -g -o bin/mcea_cpu_dbg $^ $(CFLAGS) $(LINK)

clean:
	rm -f *.o *~
	rm bin/*
