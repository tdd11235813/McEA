CC=g++
CFLAGS=-ffast-math -O3 -fopenmp
LINK=-lm
DEPS = dtlz.h util.h config.h

mcea: mcea.cpp dtlz.cpp util.cpp $(DEPS)
	@$(foreach params,$(shell find ./Makefiles -mindepth 1 -printf '%p\n'), \
		make -f $(params);)

debug: mcea.cpp dtlz.cpp util.cpp $(DEPS)
	$(CC) -g -o bin/mcea_cpu_dbg $^ $(CFLAGS) $(LINK)

clean:
	rm -f *.o *~
	rm bin/*
