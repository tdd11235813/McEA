#! /bin/bash

# run all the binaries
mkdir -p $2/population
iterations=3
counter=1
configs=`find $1/config/ -maxdepth 1 -mindepth 1 -type d`
len=`echo $configs | wc -w`
(( len = len * iterations ))

olddir=`pwd`
cd $1

for conf in $configs; do
  for run in `seq -w $iterations`; do
    bfile=${conf##*/}
    echo "[$counter/$len] run conf: ${bfile}_$run"
    javac -cp ~/bin/jMetal/jmetal-exec/target/jmetal-exec-5.4-SNAPSHOT-jar-with-dependencies.jar:. NewSPEA2Runner.java DoubleNPointCrossover.java RealUniformMutation.java $conf/Constants.java
    java -cp ~/bin/jMetal/jmetal-exec/target/jmetal-exec-5.4-SNAPSHOT-jar-with-dependencies.jar:. NewSPEA2Runner $2/population $run &> /dev/null
    (( counter+=1 ))
  done
done

cd $olddir
