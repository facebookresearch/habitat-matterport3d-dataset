#!/bin/bash
REAL_ROOT="$SAVE_DIR_PATH/real_images"
SIM_ROOT="$SAVE_DIR_PATH/simulated_images"


# Comparing simulated images with real images
for real in "mp3d_real" "gibson_real"
do
   for sim in "replica_sim" "robothor_sim" "mp3d_sim" "gibson_4_plus_sim" \
              "gibson_sim" "scannet_sim" "hm3d_sim"
   do
       echo "=========> Comparing $sim with $real"
       python measure_visual_fidelity.py \
           --real-path "$REAL_ROOT/$real" \
           --sim-path "$SIM_ROOT/$sim"
   done
done


# Comparing real images with real images
for real_1 in "mp3d_real" "gibson_real"
do
   for real_2 in "mp3d_real" "gibson_real"
   do
       echo "=========> Comparing $real_1 with $real_2"
       python measure_visual_fidelity.py \
           --real-path "$REAL_ROOT/$real_1" \
           --sim-path "$REAL_ROOT/$real_2"
   done
done