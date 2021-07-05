#!/usr/bin/env bash
today=`date '+%y%m%d'`
dirname_out=out-$today
dirname_err=err-$today
if [ ! -e $PWD/log/$dirname_out ] || [ ! -e $PWD/log/$dirname_err ]; then
    mkdir $PWD/log/$dirname_out
    mkdir $PWD/log/$dirname_err
fi
for ((i=0; i<1; i++))
do
filename=${i}_DNNProposed_synth
lambda_fair=$1
T=1000
mom=0.9
lr=0.001 
batchsize=1000 
nn_type=1
#opt="SGD" 
opt="Adam" 
mode=${filename##*_}
synth_num=${filename%%_*}

FIOflag=0
if [[ $filename =~ FIO ]] ; then
    FIOflag=1
fi
Removeflag=0
if [[ $filename =~ Remove ]] ; then
    Removeflag=1
fi
if [[ $filename =~ Logistic ]] ; then
    nn_type=2
fi
unconstr=0
if [[ $filename =~ Unconstrained ]] ; then
    unconstr=1
fi
Exflag=0
if [[ $filename =~ EX ]] ; then
    Exflag=1
fi
oracle=0
if [[ $filename =~ Oracle ]] ; then
    oracle=1
fi
uncertain="No"
if [[ $filename =~ UcA ]] ; then
    uncertain="YesA"
fi
if [[ $filename =~ UcB ]] ; then
    uncertain="YesB"
fi
if [[ $filename =~ UcAB ]] ; then
    uncertain="YesAB"
fi

echo `date '+%y/%m/%d %H:%M:%S'` $filename
nohup python run.py $lambda_fair $T $mode $lr $mom $opt $batchsize $nn_type $synth_num $FIOflag $Removeflag $unconstr $Exflag $oracle $uncertain > $PWD/log/$dirname_out/$filename-$synth_num-lamb$lambda_fair-T$T-lr$lr-mom$mom-$opt-B$batchsize-nn$nn_type.log 2> $PWD/log/$dirname_err/$filename-$synth_num-lamb$lambda_fair-T$T-lr$lr-mom$mom-$opt-B$batchsize-nn$nn_type.log < /dev/null && tput bel && sleep 1 && tput bel && sleep 1 && tput bel && echo -e `date '+%y/%m/%d %H:%M:%S'` "\n" || { RET=$?; echo "error occured" && tput bel && sleep 1 && tput bel; sleep 1; } &
done 
