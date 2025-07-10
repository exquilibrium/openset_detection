#!/bin/bash 

echo 'Testing with base weights:' $1
echo 'Num OS Classes:' $2
echo 'SaveName:' $3
echo '   Train:' $4
echo '     Val:' $5
echo '    Test:' $6

TEST="False" # EVAL_ONLY if FALSE

echo Using normal pipeline
if [ $TEST == "True" ]
then
    echo Testing data
    python test_data_yolo.py $1 $4 --num_classes $2 --saveNm $3
    python test_data_yolo.py $1 $5 --num_classes $2 --saveNm $3
    python test_data_yolo.py $1 $6 --num_classes $2 --saveNm $3

    echo Associating data
    python associate_data.py YOLOv8 --saveNm $3 --dataset custom
fi
echo Getting Results
python get_results.py YOLOv8 --saveNm $3 --dataset custom --saveResults True
