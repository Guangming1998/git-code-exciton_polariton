#!/bin/bash

for f in {3,6,9,12,15,18}
do
   mv cavitycouple_$f/CJJmol_average.csv CJJmol_eph_$f.csv
   mv cavity*_$f/CJJcav_average.csv CJJcav_eph_$f.csv
   mv cavity*_$f/CJJ_average.csv CJJ_eph_$f.csv  
done
