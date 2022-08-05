#!/usr/local/bin/bash

# allow parallel execution by terminating line w &
# using "\j", @P for num current processes requires
# bash >= 4.4
# If wanted multiple arguments passed in file, then can
# expand with e.g. ${args[@]}
params='training_params.txt'
sed '/^ *#/d;s/#.*//' $params > 'clean_params.txt'
clean_pars='clean_params.txt'
num_procs=6 # max number of runs to try at once
num_jobs="\j"  # The prompt escape for number of jobs currently running
while read -a train_pars;
do
  while ((${num_jobs@P}>=$num_procs)); do
    wait -n
  done
  python train_model.py \
        --country ${train_pars[1]} \
        --cv-type ${train_pars[2]} \
        --target ${train_pars[3]} \
        --impute ${train_pars[4]} \
        --standardise ${train_pars[5]} \
        --target-transform ${train_pars[6]} \
        --log-run \
        "${train_pars[7]}" \ # --interpretable or nothing
        "${train_pars[8]}" \ # --universal-data-only or nothing
        "${train_pars[9]}" \ # --copy-to-nbrs or nothing
#   --subsel_data         Use feature subset selection
#   --n_runs N_RUNS       Number of runs
#   --test-size TEST_SIZE
#                         Proportion of data to exclude for test evaluation, default is 0.2
#   --nfolds NFOLDS       Number of folds of training set for cross validation, default is 5
#   echo ${train_pars[@]}
#   echo ${train_pars[1]} ${train_pars[2]}
done < $clean_pars
exit 0
