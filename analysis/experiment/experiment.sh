#! /bin/bash
solution_folder="/home/est/cloud/promotion/code/McEA/analysis/solutions_dtlz/"

if [ $# -ne 3 ]; then
  echo "usage: ${0} <bin or javaconfig-folder> <out-folder> <run-script>"
  exit
fi

# test for required scripts
if [ ! -f ./bestPF.py ]; then
  echo "Please start the script from its parent folder. The script bestPF.py is required."
  exit
fi

# run the experiments
./$3 $1 $2
# correct the missing tabs for jMetal
sed -i -e 's/ /	/g' `find $2/population -name "*.obj"`

# calculate the PFs
mkdir -p $2/pareto_front
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
mkdir -p $2/metrics
counter=1
fronts=`find $2/pareto_front -name "*.pf"`
len=`echo $fronts | wc -w`

for front in $fronts; do
  infos=`echo $front | sed -e 's/pareto_front/population/' -e 's/\.pf/\.info/'`
  dtlz_num=`grep dtlz_problem $infos | sed -e 's/dtlz_problem:[[:space:]]*//'`
  runtime=`grep runtime $infos | sed -e 's/runtime:[[:space:]]*\([0-9]*\.[0-9]*\) ms/\1/'`
  ffile=${front##*/}
  ffile=${ffile%.pf}
  metric_file=$2/metrics/$ffile.metrics
  echo "[$counter/$len] calc metrics: $ffile"
  mvn -pl jmetal-exec exec:java -f ~/bin/jMetal \
    -Dexec.mainClass="org.uma.jmetal.qualityIndicator.CommandLineIndicatorRunner" \
    -Dexec.args="ALL $solution_folder/DTLZ$dtlz_num.3D.pf $front TRUE" \
    2> /dev/null | head -n18 | tail -n8 1> $metric_file
  echo "RT: $runtime" >> $metric_file

  (( counter+=1 ))
done
