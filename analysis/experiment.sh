#! /bin/bash

if [ $# -ne 2 ]; then
  echo "usage: ${0} <bin-folder> <out-folder>"
  exit
fi

# test for required scripts
if [ ! -f ./bestPF.py ]; then
  echo "Please start the script from its parent folder. The scripts bestPF.py and plot_scatter3d.py are required."
  exit
fi

# prepare folder structure
mkdir -p $2/population
mkdir -p $2/pareto_front
mkdir -p $2/visualisation
mkdir -p $2/metrics
mkdir -p $2/statistics

# run all the binaries
counter=1
binaries=`find $1 -name "mcea*"`
len=`echo $binaries | wc -w`

for bin in $binaries; do
  bfile=${bin##*/}
  echo "[$counter/$len] exec bin: $bfile"
  $bin $2/population/ &> /dev/null

  (( counter+=1 ))
done

# calculate the PFs
counter=1
results=`find $2/population -name "*.obj"`
len=`echo $results | wc -w`

for pop in $results; do
  rfile=${pop##*/}
  rfile=${rfile%.obj}
  echo "[$counter/$len] calc PF: $rfile"
  python bestPF.py $2/pareto_front/$rfile.pf $pop

  (( counter+=1 ))
done

# calculate the metrics
counter=1
fronts=`find $2/pareto_front -name "*.pf"`
len=`echo $fronts | wc -w`

for front in $fronts; do
  ffile=${front##*/}
  ffile=${ffile%.pf}
  echo "[$counter/$len] calc metrics: $ffile"
  mvn -pl jmetal-exec exec:java -f ~/bin/jMetal \
    -Dexec.mainClass="org.uma.jmetal.qualityIndicator.CommandLineIndicatorRunner" \
    -Dexec.args="ALL /home/est/cloud/promotion/code/McEA/analysis/solutions_dtlz/DTLZ7.3D.pf $front TRUE" \
    2> /dev/null | head -n18 | tail -n8 1> $2/metrics/$ffile.metrics

  (( counter+=1 ))
done