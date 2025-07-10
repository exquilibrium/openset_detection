#!/bin/bash 
cd ../../mmdetection

MODEL="frcnnCEwAnchorVocCS_xml"
SAVE_NAME="frcnn_GMMDet_Voc_xml"

./test_standard_custom.sh $MODEL $SAVE_NAME custom