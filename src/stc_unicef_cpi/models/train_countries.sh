#!/usr/local/bin/bash

# allow parallel execution by terminating line w &
# using "\j", @P for num current processes requires
# bash >= 4.4
# If wanted multiple arguments passed in file, then can
# expand with e.g. ${args[@]}
params=("nigeria","senegal","benin","togo","sierra leone","burkina faso","guinea","cameroon","liberia")
num_cores=4
num_procs=6 # max number of runs to try at once
num_jobs="\j"  # The prompt escape for number of jobs currently running
for ctry in "${params[@]}";
    do
        while ((${num_jobs@P}>=$num_procs)); do
            wait -n
        done
        python train_model.py \
                --country $ctry \
                --cv-type "spatial" \
                --impute "median" \
                --standardise "robust" \
                --log-run \
                --ncores $num_cores \
                -ip "true" \
                -univ "true" \
                -cp2nbr "true" &
# --interpretable or nothing # --universal-data-only or nothing# --copy-to-nbrs or nothing
#   --subsel_data         Use feature subset selection
#   --n_runs N_RUNS       Number of runs
#   --test-size TEST_SIZE
#                         Proportion of data to exclude for test evaluation, default is 0.2
#   --nfolds NFOLDS       Number of folds of training set for cross validation, default is 5
#   echo ${train_pars[@]}
#   echo ${train_pars[1]} ${train_pars[2]}
    done
exit 0
