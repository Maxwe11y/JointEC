#!/bin/bash

# this is my first demo
# Wei Li

dropout3=(0.4 0.5)
lr=(0.0002 0.0005)
cd /home/maxwe11y/Desktop/weili/phase3/Phase3
vardate='date'


for i in ${lr[*]}
do
    for j in ${dropout3[*]}
    do

        echo `python3 JointEC_window.py --epochs 80 --lr ${i} --dropout3 ${j} --data "../ECPEC_phase_two_xatt_0.3_0.7_relu_full.pkl" ` >> /home/maxwe11y/Desktop/weili/phase3/log_window3/"JointECW_res${i}and${j}_xatt_0.7.log" 2>&1

done
done


for i in ${lr[*]}
do
    for j in ${dropout3[*]}
    do

        echo `python3 JointEC_window.py --epochs 80 --lr ${i} --dropout3 ${j} --data "../ECPEC_phase_two_gcn_0.4_0.7_relu_full.pkl" ` >> /home/maxwe11y/Desktop/weili/phase3/log_window3/"JointECW_res${i}and${j}_gcn.log" 2>&1

done
done


for i in ${lr[*]}
do
    for j in ${dropout3[*]}
    do

        echo `python3 JointEC_window.py --epochs 80 --lr ${i} --dropout3 ${j} --data "../ECPEC_phase_two_xatt_0.3_0.8_relu_full.pkl" ` >> /home/maxwe11y/Desktop/weili/phase3/log_window3/"JointECW_res${i}and${j}_xatt_0.8.log" 2>&1

done
done