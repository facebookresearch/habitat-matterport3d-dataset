REAL_ROOT="$SAVE_DIR_PATH/real_images"

python extract_gibson_real.py --dataset-dir $GIBSON_PANO_ROOT --save-dir $REAL_ROOT/gibson_real
python extract_mp3d_real.py --dataset-dir $MP3D_PANO_ROOT --save-dir $REAL_ROOT/mp3d_real