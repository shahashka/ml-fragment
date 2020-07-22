#!/bin/bash
folders=(PLPro/1 PLPro/2 ADRP NSP15_6w01/1 3CLPro/2 NSP15_6w01/2  NSP15_6vww/1 NSP15_6vww/2 3CLPro/1 3CLPro/2 3CLPro/3 DNMT3A NSUN2)
# invalid
#DNMT1 NSP15_6w01/3 NSUN6
version=april27
for i in "${folders[@]}"
do    
    echo $i
    python generator.py -pdb $version/${i}/*.pdb -csv $version/${i}/*sorted.csv -sdf $version/${i}/*sorted.sdf -sdftop $version/${i}/*top100.sdf -state high
done
#/vol/ml/shahashka/rclone copy remote:2019-nCoV/drug-screening/RELEASES/april27/receptorsV5/ april27/receptors -P
