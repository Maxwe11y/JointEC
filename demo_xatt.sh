#!/bin/bash

# this is my first demo
# Wei Li

#dropout3=(0.4)
chunk_size=(2)
tf=(0.5 1)
cd /home/maxwe11y/Desktop/weili/phase3/Phase3
vardate='date'
for i in ${chunk_size[*]}
do
    for j in ${tf[*]}
    do

        echo `python3 JointEC_window.py --epochs 100 --chunk_size ${i} --tf ${j} --lr 0.001 ` >> /home/maxwe11y/Desktop/weili/phase3/case_study/"JointECW_tf_res${i}and${j}.log" 2>&1

done
done


for i in ${chunk_size[*]}
do
    for j in ${tf[*]}
    do

        echo `python3 JointEC.py --epochs 100 --chunk_size ${i} --tf ${j} --lr 0.0005 ` >> /home/maxwe11y/Desktop/weili/phase3/case_study/"JointEC_tf_res${i}and${j}.log" 2>&1

done
done