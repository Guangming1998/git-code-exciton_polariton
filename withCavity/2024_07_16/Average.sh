#!/bin/bash

for f in {2..8}
do
  cp CJJ_average.py CJJcav_average.py CJJmol_average.py *couple_$f/
  cd *couple_$f/
  python CJJ_average.py
  python CJJcav_average.py
  python CJJmol_average.py
  cd ..
done
