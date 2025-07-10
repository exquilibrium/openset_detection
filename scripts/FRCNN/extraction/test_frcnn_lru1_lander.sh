#!/bin/bash 
cd ../../mmdetection

MODEL="frcnnCEwAnchorVocCS_lru1_lander"
SAVE_NAME="frcnn_GMMDet_Voc_lru1_lander"

./test_standard_custom.sh $MODEL $SAVE_NAME custom