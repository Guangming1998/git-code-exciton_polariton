#!/bin/bash

for f in {5..6}
do 
  mkdir cavitycouple_$f
  cp cavity.py cavity.bash Cavity_ssh.py jobarray.sh data_average.py final_aggregation.sh cavitycouple_$f
  cd cavitycouple_$f
  sed -i '86s/1/'$f'/' cavity.py
  sed -i '2s/1/'$f'/' jobarray.sh
  cd ..
done

for f in {5..6}
do
  cd cavitycouple_$f
  for i in {1..1000}
  do 
    mkdir tr$i
    cp cavity.py cavity.bash Cavity_ssh.py tr$i
  done
  cd ..
done


