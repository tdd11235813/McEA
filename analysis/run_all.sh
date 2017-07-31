#! /bin/bash

./experiment.sh ../cpu/bin/ ~/data/mcea_test/cpu_testrun_30 run_bin.sh
./experiment.sh ../gpu/bin/ ~/data/mcea_test/gpu_testrun_30 run_bin.sh
./experiment.sh ../compareAlg ~/data/mcea_test/nsgaii_testrun_30 run_nsgaii.sh
./experiment.sh ../compareAlg ~/data/mcea_test/spea2_testrun_30 run_spea2.sh

# calculate statistics - not complete
Rscript analysis.R ~/data/mcea_test/cpu_testrun_30/metrics ~/data/mcea_test/pu_testrun_30/metrics
