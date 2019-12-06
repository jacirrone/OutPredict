"""
Run Class Implementation

"""


from Dataset import Dataset
from Model import Model
import numpy as np
import os, inspect, gc

class Run:
    def __init__(self):
        pass
        #print("Constructor Run")

        # import sys, site
        # #sys.path.remove(site.USER_SITE)
        # psys_arr = []
        # for psys in sys.path:
        #     if psys.__contains__(site.USER_SITE):
        #         psys_arr.append(psys)
        # for psys_to_remove in psys_arr:
        #     sys.path.remove(psys_to_remove)
        # print('\n'.join(sys.path))


    def launch_out_predict(self, ml_model, tree_method, k, ntrees, tau, timehorizon, datasetname, test_size, prior_file, prior_type, datatype, data_type_lo, bias_score_splitnodeFeature, thres_coeff_var, num_ets_lo, time_step, tuning_genebygene_randomized, n_iter_search, tfa_bool, num_threads, save_models, name_run, flag_print, parse_4dyng3, poot, auto_meth, rnd_seed):

        np.random.seed(rnd_seed)

        ds1 = None
        gc.collect()

        if ml_model == "RF":
            if poot:
                print("Loading and Preprocessing Dataset ", datasetname+" ...\n")
            ds1 = Dataset(datasetname, rnd_seed)

            if prior_file == "no":
                name_run = datasetname + "_output_" + tree_method + "_K" + k + "_ntrees" + str(ntrees) + "_datatype" + str(datatype) + "_LOdata" + str(
                    data_type_lo)
            else:
                name_run = datasetname + "_output_" + tree_method + "_K" + k + "_ntrees" + str(ntrees) + "_datatype" + str(datatype) + "_LOdata" + str(
                    data_type_lo) + "_prior" + (
                           (prior_file).split('.')[0])

            #h1 and _bias_score_splitnodeFeature removed
            # if prior_file == "no":
            #     name_run = datasetname + "_output_" + tree_method + "_K" + k + "_ntrees" + str(ntrees) + "_h" + str(
            #         timehorizon) + "_datatype" + str(datatype) + "_LOdata" + str(
            #         data_type_lo) + "_bias_score_splitnodeFeature" + str(bias_score_splitnodeFeature)
            # else:
            #     name_run = datasetname + "_output_" + tree_method + "_K" + k + "_ntrees" + str(ntrees) + "_h" + str(
            #         timehorizon) + "_datatype" + str(datatype) + "_LOdata" + str(
            #         data_type_lo) + "_bias_score_splitnodeFeature" + str(bias_score_splitnodeFeature) + "_prior" + (
            #                (prior_file).split('.')[0])

            if datasetname == "bsubtilis-inf-ng":
                delTmax = 100000
                delTmin = 0
            elif datasetname == "ecoli-inf15":
                delTmax = 100000
                delTmin = 0
            elif datasetname == "dream4-inf-ng" or datasetname=="dream10_debug":
                delTmax = 100000  # 110
                delTmin = 0
                # tau = 45
            elif datasetname == "yeast-inf-ng":
                delTmax = 100000  # 110#5000
                delTmin = 0  # 4
                # tau = 45#15
            elif datasetname == "bsubtilis2-inf-ng":
                delTmax = 100000  # 110#60
                delTmin = 0  # 15
                # tau = 45#15
            else: #datasetname == "drosophila-dynGen":
                delTmax = 100000
                delTmin = 0

        #Name_run commented because now outpredict does everything automatically
        # if num_ets_lo > 0:
        #     name_run = name_run + "_num_ets_lo" + str(num_ets_lo)
        #
        # if not (tfa_bool):
        #     name_run = name_run + "_NO_TFA"
        # else:
        #     name_run = name_run + "_TFA"
        #
        # if tuning_genebygene_randomized:
        #     name_run = name_run + "_RandSearOpt" + str(n_iter_search)
        #
        # if time_step:
        #     name_run = name_run + "_TimeStep"
        # else:
        #     name_run = name_run + "_ODE_PredWlogDiff"

        script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory

        path_dataset = script_dir + "/Datasets/" + datasetname + "/"

        ds1.flag_print = flag_print
        ds1.parse_4dyng3 = parse_4dyng3
        ds1.auto_meth = auto_meth
        ds1.poot = poot

        X, y, genelist, tflist, goldstandard, output_path, priors_data, X_test_ss, X_test_ts, y_test_ss, y_test_ts, x_test_ts_current_timepoint, y_test_ts_future_timepoint, deltas, x_test_ts_timepoint0, index_steady_state_new, index_time_points_new, design_mat, delta_vect, res_mat2 = ds1.loadData(
            path_dataset, name_run, script_dir, datatype, data_type_lo, delTmax, delTmin, tau, tfa_bool,
            timehorizon, test_size, num_ets_lo, time_step, thres_coeff_var, prior_type, prior_file)

        if prior_file == "no":
            if os.path.exists(output_path + "/priors"):
                os.rmdir(output_path + "/priors")

        if np.sum(np.array(deltas)<=1)>0:
            deltas = np.asarray(deltas) + 1
        deltas = np.log(deltas)

        if flag_print:
            print("Start of modeling for tau value: ", tau)
        model_instance = Model(datatype, data_type_lo, genelist, tflist, prior_file, prior_type, script_dir,
                               path_dataset, bias_score_splitnodeFeature, X_test_ss, X_test_ts, y_test_ss,
                               y_test_ts, x_test_ts_current_timepoint, y_test_ts_future_timepoint, deltas, tau,
                               x_test_ts_timepoint0, flag_print, poot, rnd_seed)

        if ml_model == "RF":
            nthreads = num_threads
            oob_mse_avg, oob_score_avg, aupr, mse_test_future_timepoint, mean_corr_test_future_timepoint, print_out_string_to_ret, confidences, gs, outfile = model_instance.build_RF_model(
                nthreads, X, y, tree_method, k, ntrees, timehorizon, datasetname, output_path, name_run, num_ets_lo,
                priors_data, tuning_genebygene_randomized, time_step, n_iter_search, save_models,
                index_steady_state_new, index_time_points_new, design_mat, delta_vect, res_mat2, auto_meth, gs=goldstandard)

        return output_path, oob_score_avg, oob_mse_avg, aupr, mse_test_future_timepoint, mean_corr_test_future_timepoint, print_out_string_to_ret, confidences, gs, outfile
