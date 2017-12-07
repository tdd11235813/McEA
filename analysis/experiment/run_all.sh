#! /bin/bash

./experiment.sh ../../build/release/cpu/ ~/data/mcea_test/cpu_testrun_30 run_bin.sh
./experiment.sh ../../build/release/gpu/ ~/data/mcea_test/gpu_testrun_30 run_bin.sh
./experiment.sh ../compareAlg ~/data/mcea_test/nsgaii_testrun_30 run_nsgaii.sh
./experiment.sh ../compareAlg ~/data/mcea_test/spea2_testrun_30 run_spea2.sh

# calculate statistics - not complete
Rscript ../statistic/analysis.R ~/data/mcea_test/cpu_testrun_30/metrics ~/data/mcea_test/gpu_testrun_30/metrics
