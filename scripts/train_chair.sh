#! /bin/bash

###################################### config #######################################################
SCENE=chair
BATCH_SIZE=256
PATCH_SIZE=64
PATCH_STRIDE=6
EXPNAME=${SCENE}_B${BATCH_SIZE}_P${PATCH_SIZE}_PS${PATCH_STRIDE}_1corr0.18,1,0.46,1_0.01geoCorr0.25,1,1,1
# EXPNAME=${SCENE}_tmp
mkdir -p logs/$EXPNAME

python -u run_nerf.py --gpuid 1 \
--expname ${EXPNAME}  \
--config configs/${SCENE}.txt  \
--i_print 200  \
--i_verbose 200  \
--i_testset 200 \
--i_video 5000  \
--i_img 100 \
--i_weights 2500 \
--max_steps 50001 \
--batch_size $BATCH_SIZE \
--patch_size $PATCH_SIZE  \
--patch_stride  $PATCH_STRIDE   \
--app_corr_params 0.18 1 0.46 1 \
--geo_corr_params 0.25 1 1 1 \
--fast_mode \
--no_semantics
