

from OutPredict import OutPredict

if __name__ == '__main__':

    op = OutPredict()

    op.num_of_trees = 300  # The number of Trees for Random Forests

    op.input_dir_name = "plant_ds"  # Name of Directory, inside OP_3/Datasets/, containing the dataset

    op.test_set_split_ratio = 0.15  # The percentage of data points to use for the test set separately for time-series and steady-state, e.g. 0.15, 15% of steady-state data will be used as test set, 15% of the time-series data (last time points of time-series)

    op.training_data_type = "TS-SS"  # whether to use for training TS(time-series), SS(steady-stae) or TS-SS (time-series and steady-state)
    op.leave_out_data_type = "TS"  # whether to use for training TS(time-series), SS(steady-stae) or TS-SS (time-series and steady-state)

    op.genes_coeff_of_var_threshold = 0  # coefficient of variance threshold to filter the genes to modeling; 0 to modeling all genes

    op.num_of_cores = 20  # (Integer) number of cores to use for parallelization

    #op.prior_file_name = "gold_standard.tsv" #either name of file containing prior knowledge or empty

    #op.priors = "gold_standard" #"steady_state"  # gold_standard or steady_state or empty

    op.run()
