#!/bin/bash

# this is my first demo
# Wei Li

dropout=(0.2 0.3 0.4 0.5 0.6 0.7 0.8)
chunk_size=(5 10 15 20)
cd /home/maxwe11y/Desktop/weili/phase3/Phase3
vardate='date'
for i in ${dropout[*]}
do
    for j in ${chunk_size[*]}
    do
        #echo "hello world: ${i}${j}"
        echo `python3 JointEC.py --epochs 50 --dropout ${i} --chunk_size ${j} --lr 0.0005 ` >> /home/maxwe11y/Desktop/weili/phase3/log_p3/"res${i}and${j}.log" 2>&1
        #echo ${i}${j}
done
done
