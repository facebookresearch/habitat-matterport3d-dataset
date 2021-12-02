GIBSON_DIR="<PATH to raw Gibson dataset path>"
MP3D_PANO_DIR="<PATH to panoramas rendered from MP3D>"

SAVE_DIR_PATH="<DIRECTORY TO SAVE extracted images>"
SAVE_ROOT="$SAVE_DIR_PATH/real_images"

python extract_gibson_real.py --dataset-dir $GIBSON_DIR --save-dir $SAVE_ROOT/gibson_real
python extract_mp3d_real.py --dataset-dir $MP3D_PANO_DIR --save-dir $SAVE_ROOT/mp3d_real