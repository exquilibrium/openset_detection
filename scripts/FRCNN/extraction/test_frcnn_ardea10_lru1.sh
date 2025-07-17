#!/bin/bash 
cd ../../mmdetection

MODEL="frcnnCEwAnchorVocCS_ardea10_lru1"
SAVE_NAME="frcnn_GMMDet_Voc_ardea10_lru1"

./test_standard_custom.sh $MODEL $SAVE_NAME custom