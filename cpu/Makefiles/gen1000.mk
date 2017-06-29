CC=g++
CFLAGS=-ffast-math -O3 -fopenmp
LINK=-lm
DEPS = dtlz.h util.h config.h

GENERATIONS=1000
POP_WIDTH=400
PARAMS=20
NRAD=2
THREADS=8
VADS_SCALE=100
CONFIG=_g$(GENERATIONS)_pw$(POP_WIDTH)_p$(PARAMS)_r$(NRAD)_t$(THREADS)_vs$(VADS_SCALE)
OUTFILE='"out$(CONFIG).obj"'


mcea_cpu: mcea.cpp dtlz.cpp util.cpp $(DEPS)
	$(CC) -o bin/$@$(CONFIG) $^ $(CFLAGS) $(LINK) \
		-DGENERATIONS=$(GENERATIONS) \
		-DPOP_WIDTH=$(POP_WIDTH) \
		-DPARAMS=$(PARAMS) \
		-DN_RAD=$(NRAD) \
		-DTHREADS=$(THREADS) \
		-DVADS_SCALE=$(VADS_SCALE) \
		-DOUTFILE=$(OUTFILE)
