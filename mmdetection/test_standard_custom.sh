#!/bin/bash 

echo 'Testing with base weights:' $1
echo 'Save name:' $2
echo 'Dataset:' $3

TEST="False" # EVAL_ONLY if FALSE
CUSTOM="True"

if [ $CUSTOM == "True" ]
then
    echo Using custom pipeline
    if [ $TEST == "True" ]
    then
        echo Testing data
        python test_data_custom.py --subset train --dir "$1" --saveNm "$2" --dataset "$3" # CS
        python test_data_custom.py --subset val --dir "$1" --saveNm "$2" --dataset "$3" # CS
        python test_data_custom.py --subset test --dir "$1" --saveNm "$2" --dataset "$3" # OS

        echo Associating data
        python associate_data_custom.py FRCNN --saveNm "$2" --dataset "$3"
    fi
    echo Getting Results
    python get_results_custom.py FRCNN --saveNm "$2" --dataset "$3" --saveResults True
else
    echo Using default pipeline
    if [ $TEST == "True" ]
    then
        echo Testing data
        python test_data.py --subset train --dir "$1" --saveNm "$2" --dataset "$3" # CS
        python test_data.py --subset val --dir "$1" --saveNm "$2" --dataset "$3" # CS
        python test_data.py --subset test --dir "$1" --saveNm "$2" --dataset "$3" # OS

        echo Associating data
        python associate_data.py FRCNN --saveNm "$2" --dataset "$3"
    fi
    echo Getting Results
    python get_results.py FRCNN --saveNm "$2" --dataset "$3" --saveResults True
fi
