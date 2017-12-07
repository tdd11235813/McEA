#! /bin/bash

# run all the binaries
mkdir -p $2/population
iterations=15
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

