#! /bin/bash
solution_folder="/home/est/cloud/promotion/code/McEA/analysis/solutions_dtlz/"

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
iterations=30
counter=1
binaries=`find $1 -name "mcea*"`
len=`echo $binaries | wc -w`
(( len = len * iterations ))

for bin in $binaries; do
  for run in `seq -w $iterations`; do
    bfile=${bin##*/}
    echo "[$counter/$len] exec bin: ${bfile}_$run"
    $bin $2/population/ $run &> /dev/null
    (( counter+=1 ))
  done
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
  infos=`echo $front | sed -e 's/pareto_front/population/' -e 's/\.pf/\.info/'`
  dtlz_num=`grep dtlz_problem $infos | sed -e 's/dtlz_problem:[[:space:]]*//'`
  ffile=${front##*/}
  ffile=${ffile%.pf}
  echo "[$counter/$len] calc metrics: $ffile"
  mvn -pl jmetal-exec exec:java -f ~/bin/jMetal \
    -Dexec.mainClass="org.uma.jmetal.qualityIndicator.CommandLineIndicatorRunner" \
    -Dexec.args="ALL $solution_folder/DTLZ$dtlz_num.3D.pf $front TRUE" \
    2> /dev/null | head -n18 | tail -n8 1> $2/metrics/$ffile.metrics

  (( counter+=1 ))
done
