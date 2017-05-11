CC=gcc
CFLAGS=-ffast-math -O3
LINK=-lm
DEPS = dtlz.h util.h

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

mcea_cpu: util.o dtlz.o mcea.o
	$(CC) -o $@ $^ $(CFLAGS) $(LINK)

debug: mcea.cpp dtlz.cpp util.cpp
	$(CC) -g -o mcea_cpu_dbg $^ $(CFLAGS) $(LINK)

clean:
	rm -f *.o *~
