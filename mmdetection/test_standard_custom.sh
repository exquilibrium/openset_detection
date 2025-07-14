#!/bin/bash 

echo 'Testing with base weights:' $1
echo 'Save name:' $2
echo 'Dataset:' $3

TEST="True" # EVAL_ONLY if FALSE

echo Using default pipeline
if [ $TEST == "True" ]
then
    echo Testing data
    #python test_data.py --subset train --dir "$1" --saveNm "$2" --dataset "$3" # CS, Mahalanobis++
    #python test_data.py --subset val --dir "$1" --saveNm "$2" --dataset "$3" # CS, Mahalanobis++
    #python test_data.py --subset test --dir "$1" --saveNm "$2" --dataset "$3" # OS
    #python test_data.py --subset testOOD --dir "$1" --saveNm "$2" --dataset "$3" # OOD, Mahalanobis++

    echo Associating data
    #python associate_data.py FRCNN --saveNm "$2" --dataset "$3"
fi
echo Getting Results
#python get_results.py FRCNN --saveNm "$2" --dataset "$3" --saveResults True
python get_results_mahalanobis.py --saveNm "$2" --dataset "$3"


