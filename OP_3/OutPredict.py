"""
% Copyright (C) 2019 Jacopo Cirrone

OutPredict Main Class Implementation

"""




# from numpy import load
# import pandas as pd
# import time
# from operator import itemgetter
import matplotlib.pyplot as plt
# from numpy.random import *
# import re
# import os, pwd
#from numpy import *
#import time


import inspect
import os
import gc
import argparse as argp
from Run import Run
import sys, site
import csv
from network_inference import NetworkInference

plt.switch_backend('agg')

class OutPredict(object):


    def __init__(self):
        #print("Constructor Module")
        #ntrees
        self.num_of_trees = None
        #datasetname\
        self.input_dir_name = None  # name of folder containing the dataset
        #test_size \
        self.test_set_split_ratio = None# ratio e.g. 0.15
        #prior_file\
        self.prior_file_name = None # name of file or no
        #prior_type \
        self.priors = None #gold_standard or steady state or no # binary_all, real_all or no
        #datatype \
        self.training_data_type= None # TS, SS or TS-SS
        #data_type_lo \
        self.leave_out_data_type= None # TS, SS or TS-SS
        #thres_coeff_var \
        self.genes_coeff_of_var_threshold = None #0 to model all genes
        #time_step  \   # [default is auto_meth=True so it learns the best based on oob]
        self.time_step_or_ode_log = None # 1 for time-step, 2 for ode-log # which method to use; either time-step or ode-log
        #num_threads \
        self.num_of_cores = None # Integer

        # import sys, site
        # # sys.path.remove(site.USER_SITE)
        # psys_arr = []
        # for psys in sys.path:
        #     if psys.__contains__(site.USER_SITE):
        #         psys_arr.append(psys)
        # for psys_to_remove in psys_arr:
        #     sys.path.remove(psys_to_remove)
        # print('\n'.join(sys.path))



    def call_run_obj(self, ml_model, *args):

        import numpy as np

        slurm_job = args[0]

        if slurm_job:
            job_id = slurm_job.split("-")[1]
        else:
            job_id = slurm_job

        timehorizon = args[4]

        datasetname = args[5]

        test_size = args[6]

        prior_file = args[7]

        prior_type = args[8]

        datatype = args[9]

        data_type_lo = args[10]

        bias_score_splitnodeFeature = args[11]

        thres_coeff_var = args[12]

        num_ets_lo = args[13]

        time_step = args[14]

        tuning_genebygene_randomized = args[15]

        n_iter_search = args[16]

        tfa_bool = args[17]

        num_threads = args[18]

        save_models = args[19]

        flag_print = args[20]

        parse_4dyng3 = args[21]

        auto_meth = args[22]

        rnd_seed = args[23]

        if self.input_dir_name:
            script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            path_dataset = script_dir + "/Datasets/" + self.input_dir_name + "/"

            if not(os.path.isdir(path_dataset)):
                print("Dataset  "+path_dataset+"  does NOT exist")
                exit(1)
            else:
                datasetname = self.input_dir_name
        if self.test_set_split_ratio:
            if not(self.is_number(self.test_set_split_ratio)):
                print("test_set_split_ratio is NOT a number")
                exit(1)
            else:
                test_size = self.test_set_split_ratio
        if self.prior_file_name:
            prior_file = self.prior_file_name
            tree_method = "RF-mod"
            if self.prior_file_name == "":
                prior_file = "no"
                prior_type = "no"
                tree_method = "RF"
                if sys.path.__contains__(site.USER_SITE):
                    print("Error ***************************************** Error ")
                    print("To run OutPredict WITHOUT priors please run:")
                    print("python -s pipeline_new_organism.py")
                    print("and do NOT set the params prior_file_name and priors")
                    print("*****************************************")
                    print("To run OutPredict WITH priors please run:")
                    print("python pipeline_new_organism.py")
                    print("After properly setting both the params prior_file_name and priors")
                    exit(1)
            else:
                if self.priors:
                    tree_method = "RF-mod"
                    if self.priors != "gold_standard" and self.priors != "steady_state" and self.priors != "" and self.priors != "no":
                        print("*****************************************")
                        print("Error: priors param value has to be either gold_standard or steady_state")
                        exit(1)
                    else:
                        if self.priors == "gold_standard":
                            prior_type = "binary_all"
                            if not(os.path.exists(path_dataset+self.prior_file_name)):
                                print(FileNotFoundError("gold standard file "+path_dataset+self.prior_file_name+" does NOT exist"))
                                exit(1)
                        elif self.priors == "steady_state":
                            prior_type = "real_all"
                        elif self.priors == "" or self.priors == "no":
                            prior_type = "no"
                            prior_file = "no"
                            tree_method = "RF"
                            if sys.path.__contains__(site.USER_SITE):
                                print("Error ***************************************** Error ")
                                print("To run OutPredict WITHOUT priors please run:")
                                print("python -s pipeline_new_organism.py")
                                print("and do NOT set the params prior_file_name and priors")
                                print("*****************************************")
                                print("To run OutPredict WITH priors please run:")
                                print("python pipeline_new_organism.py")
                                print("After properly setting both the params prior_file_name and priors")
                                exit(1)
                else:
                    if sys.path.__contains__(site.USER_SITE):
                        print("Error **************************************** Error ")
                        print("To run OutPredict WITHOUT priors please run:")
                        print("python -s pipeline_new_organism.py")
                        print("and do NOT set the params prior_file_name and priors")
                        print("*****************************************")
                        print("To run OutPredict WITH priors please run:")
                        print("python pipeline_new_organism.py")
                        print("After properly setting both the params prior_file_name and priors")
                        exit(1)
        else:
            prior_type = "no"
            prior_file = "no"
            tree_method = "RF"
            if sys.path.__contains__(site.USER_SITE):
                print("Error ***************************************** Error ")
                print("To run OutPredict WITHOUT priors please run:")
                print("python -s pipeline_new_organism.py")
                print("and do NOT set the params prior_file_name and priors")
                print("*****************************************")
                print("To run OutPredict WITH priors please run:")
                print("python pipeline_new_organism.py")
                print("After properly setting both the params prior_file_name and priors")
                exit(1)
        if self.training_data_type:
            if self.training_data_type!="TS-SS" and self.training_data_type!="TS" and self.training_data_type!="SS":
                print("*****************************************")
                print("Error: training_data_type param value has to be TS-SS, TS, or SS")
                exit(1)
            else:
                datatype = self.training_data_type
        if self.leave_out_data_type:
            if self.leave_out_data_type!="TS-SS" and self.leave_out_data_type!="TS" and self.leave_out_data_type!="SS":
                print("*****************************************")
                print("Error: leave_out_data_type param value has to be TS-SS, TS, or SS")
                exit(1)
            else:
                data_type_lo = self.leave_out_data_type
        if self.genes_coeff_of_var_threshold:
            if not(self.is_number(self.genes_coeff_of_var_threshold)):
                print("test_set_split_ratio is NOT a number")
                exit(1)
            else:
                thres_coeff_var = self.genes_coeff_of_var_threshold
        if self.time_step_or_ode_log:
            if self.time_step_or_ode_log!="time-step" and self.time_step_or_ode_log!="ode-log":
                print("*****************************************")
                print("Error: time_step_or_ode_log param value has to be either time-step, or ode-log")
                exit(1)
            else:
                if self.time_step_or_ode_log == "time-step":
                    time_step = "True"
                elif self.time_step_or_ode_log == "ode-log":
                    time_step = "False"
        if self.num_of_cores:
            if not(self.is_number(self.num_of_cores)):
                print("num_of_cores is NOT a number")
                exit(1)
            else:
                num_threads = self.num_of_cores


        name_run = ""

        # if ml_model == "RF":
        # tree_method = args[1]
        k = args[2]
        ntrees = args[3]

        if self.num_of_trees:
            if not(self.is_number(self.num_of_trees)):
                print("num_of_trees is NOT a number")
                exit(1)
            else:
                ntrees = self.num_of_trees

        # #Check here the conda environment
        # conda_env = os.environ['CONDA_DEFAULT_ENV']
        # if conda_env == "op3":
        #     tree_method = "RF"
        #     # sys.path.insert(0,"//anaconda/envs/op3/lib/python3.7/site-packages")
        # elif conda_env == "op3_priors":
        #     tree_method = "RF-mod"
        # else:
        #     print("The conda environment has not been set up correctly")
        #     exit(1)

        if tree_method == "RF":
            print(":::::::RUNNING OUTPREDICT WITHOUT PRIORS:::::::::")
            if sys.path.__contains__(site.USER_SITE):
                print("Error***************************************** Error ")
                print("To run OutPredict WITHOUT priors please run:")
                print("python -s pipeline_new_organism.py")
                print("and do NOT set the params prior_file_name and priors")
                print("*****************************************")
                print("To run OutPredict WITH priors please run:")
                print("python pipeline_new_organism.py")
                print("After properly setting both the params prior_file_name and priors")
                exit(1)
        if tree_method == "RF-mod":
            print(":::::::RUNNING OUTPREDICT WITH PRIORS:::::::::")
            if not(sys.path.__contains__(site.USER_SITE)):
                print("Error ***************************************** Error ")
                print("To run OutPredict WITH priors please run:")
                print("python pipeline_new_organism.py")
                print("After properly setting both the params prior_file_name and priors")
                print("*****************************************")
                print("To run OutPredict WITHOUT priors please run:")
                print("python -s pipeline_new_organism.py")
                print("and do NOT set the params prior_file_name and priors")
                exit(1)


        time_step = self.str_to_bool(time_step)
        tuning_genebygene_randomized = self.str_to_bool(tuning_genebygene_randomized)
        tfa_bool = self.str_to_bool(tfa_bool)
        flag_print = self.str_to_bool(flag_print)
        parse_4dyng3 = self.str_to_bool(parse_4dyng3)
        auto_meth = self.str_to_bool(auto_meth)

        if not(time_step):
            tau_vect = [5, 1, 45]
        else:
            tau_vect = [0]

        if auto_meth:
            tau_vect = [0, 5, 1, 45] # tau=0 implies to run time-step


        oob_score_avg_vect = np.ones(len(tau_vect))
        oob_mse_avg_vect = np.ones(len(tau_vect))
        aupr_vect = np.ones(len(tau_vect))
        mse_test_future_timepoint_vect = np.ones(len(tau_vect))
        mean_corr_test_future_timepoint_vect = np.ones(len(tau_vect))
        print_out_vect = [""]*len(tau_vect)
        confidences_vect = []

        print_only_one_time = True

        for run_i, tau in enumerate(tau_vect):
            gc.collect()
            run_obj = Run()

            if auto_meth:
                if tau==0:
                    time_step = True
                else:
                    time_step = False

            if flag_print:
                print(
                    "slurm_job, tree_method, max_feat, k, ntrees, ntrees, timehorizon, datasetname, test_size, prior_file, prior_type, datatype, data_type_lo, bias_score_splitnodeFeature, thres_coeff_var, num_ets_lo, time_step, tuning_genebygene_randomized, n_iter_search, tfa_bool, num_threads, save_models, flag_print, auto_meth")
                print(slurm_job, tree_method, "max_feat", k, "ntrees", ntrees, timehorizon, datasetname, test_size,
                      prior_file, prior_type, datatype, data_type_lo, bias_score_splitnodeFeature, thres_coeff_var,
                      num_ets_lo, time_step, tuning_genebygene_randomized, n_iter_search, tfa_bool, num_threads,
                      save_models, flag_print, auto_meth, rnd_seed)

            output_path, oob_score_avg, oob_mse_avg, aupr, mse_test_future_timepoint, mean_corr_test_future_timepoint, print_out_string_to_ret, confidences, gs, outfile = run_obj.launch_out_predict(ml_model, tree_method, k, ntrees, tau, timehorizon, datasetname, test_size, prior_file, prior_type, datatype, data_type_lo, bias_score_splitnodeFeature, thres_coeff_var, num_ets_lo, time_step, tuning_genebygene_randomized, n_iter_search, tfa_bool, num_threads, save_models, name_run, flag_print, parse_4dyng3, print_only_one_time, auto_meth, rnd_seed)

            if flag_print:
                outfile.write(
                    "slurm_job, tree_method, max_feat, k, ntrees, ntrees, timehorizon, datasetname, test_size, prior_file, prior_type, datatype, data_type_lo, bias_score_splitnodeFeature, thres_coeff_var, num_ets_lo, time_step, tuning_genebygene_randomized, n_iter_search, tfa_bool, num_threads, save_models, flag_print, auto_meth")
                outfile.write(str(slurm_job)+str(tree_method)+str("max_feat")+str(k)+str("ntrees")+str(ntrees)+
                              str(timehorizon)+str(datasetname)+str(test_size)+str(prior_file)+str(prior_type)+str(datatype)+
                              str(data_type_lo)+str(bias_score_splitnodeFeature)+str(thres_coeff_var)+str(num_ets_lo)+
                              str(time_step)+str(tuning_genebygene_randomized)+str(n_iter_search)+
                              str(tfa_bool)+str(num_threads)+str(save_models)+str(flag_print)+str(auto_meth)+str(rnd_seed))
                print("End of modeling with tau value: ", tau, ". The oob_score_avg is: ", oob_score_avg)
                print("End of modeling with tau value: ", tau, ". The oob_mse_avg is: ", oob_mse_avg)
            oob_score_avg_vect[run_i] = oob_score_avg
            oob_mse_avg_vect[run_i] = oob_mse_avg
            aupr_vect[run_i] = aupr
            mse_test_future_timepoint_vect[run_i] = mse_test_future_timepoint
            mean_corr_test_future_timepoint_vect[run_i] = mean_corr_test_future_timepoint
            print_out_vect[run_i] = print_out_string_to_ret
            confidences_vect.append(confidences)

            if auto_meth:
                print_only_one_time = False

            if flag_print:
                print("\n \n \n \n \n")

        for run_i, tau in enumerate(tau_vect):
            if flag_print:
                print("Modeling for tau value: ", tau)
                print("The oob_score_avg is: ", oob_score_avg_vect[run_i])
                print("The oob_mse_avg is: ", oob_mse_avg_vect[run_i])
                print("The aupr is: ", aupr_vect[run_i])
                print("The mse_test_set is: ", mse_test_future_timepoint_vect[run_i])
                print("The mean_corr_test_set is: ", mean_corr_test_future_timepoint_vect[run_i])

        if auto_meth:
            index_best = np.where(oob_mse_avg_vect == np.min(oob_mse_avg_vect))[0][0]
            print("OutPredict has found the best model according to the out-of-bag score... \n")
            print("The best model is compared to the Penultimate Value Naive approach. \n")


            confidences = confidences_vect[index_best]

            net_inf = NetworkInference(flag_print, rnd_seed)
            aupr, random_aupr = net_inf.summarize_results(output_path, "PR_CurveBestModel", confidences, gs, True)

            if flag_print:
                print("Area under Precision-Recall based on goldstandard: ", aupr)

                outfile.write("Area under Precision-Recall based on goldstandard: " + str(aupr) + "\n")

                outfile.write("Random AUPR: " + str(random_aupr) + "\n")

            confidences.to_csv(output_path + "/Matrix_TF_gene_best_model.tsv", sep="\t")
            # Print ranked list of edges
            List = [('TF', 'Target', 'Importance')]
            for source in confidences.columns.values:
                for target in confidences.index.values:
                    List.append((source, target, confidences[source][target]))
            with open(output_path + "/Ranked_list_TF_gene_best_model.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(List)

            print(print_out_vect[index_best])
            outfile.write(print_out_vect[index_best])

            outfile.write("Run_name: "+str(self.num_of_trees)+self.input_dir_name+str(self.test_set_split_ratio)+self.training_data_type+self.leave_out_data_type+str(self.genes_coeff_of_var_threshold)+str(self.num_of_cores))
            outfile.close()

            # print("The oob_score_avg of the best model is: ", oob_score_avg_vect[index_best])
            # print("The oob_mse_avg of the best model is: ", oob_mse_avg_vect[index_best])
            # print("The aupr of the best model is: ", aupr_vect[index_best])
            # print("The mse_test_set of the best model is: ", mse_test_future_timepoint_vect[index_best])
            # print("The mean_corr_test_set of the best model is: ", mean_corr_test_future_timepoint_vect[index_best])


        # # Move slurm output file to output directory
        # import pwd
        # getlogin = lambda: pwd.getpwuid(os.getuid())[0]
        # default_username = getlogin()
        # if default_username == "jc3832":
        # #if job_id:
        #     num_slurm_files = len([name for name in os.listdir(output_path + "/") if
        #                            os.path.isfile(os.path.join(output_path, name)) and "slurm" in name])
        #     os.rename("slurm-" + job_id + ".out",
        #               output_path + "/slurm_" + str(num_slurm_files) + "_" + job_id + ".txt")

    # if __name__ == '__main__':
    def run(self):
        import warnings

        warnings.filterwarnings("ignore")

        parser = argp.ArgumentParser()

        parser.add_argument('-job', '--slurm_job', type=str, help='SLURM JOB-ID', default='')

        parser.add_argument('-tm', '--tree_method', type=str, help='Tree-based method: RF-Random Forest or ET-Extra Trees)', default='RF')

        parser.add_argument("-k", "--K", type=str, help="The max num of features used for node splitting", default='sqrt')

        parser.add_argument("-nt", "--ntrees", type=int, help="The number of trees", default=3)

        parser.add_argument("-th", "--timehorizon", type=int, help="The time-lag h", default=1)

        parser.add_argument("-ds", "--datasetname", type=str, help="The dataset name. e.g. bsubtilis", default='dream10')

        parser.add_argument("-ts", "--test_size", type=float, help="The size of the dataset to use as test set, leave-out points for SS and TS separately is equal to this percentage", default=0.15)

        parser.add_argument("-netslo", "--num_ets_lo", type=int, help="Number of entire time-series to Leave-out", default=0)

        parser.add_argument("-dt", "--data_type", type=str, help="Type of Data: TS, SS or TS-SS", default="TS-SS")

        parser.add_argument("-dtlo", "--data_type_lo", type=str, help="Type of Data to Leave-out: TS, SS or TS-SS", default="TS")

        parser.add_argument("-pf", "--prior_file", type=str, help="Prior file that will be read from directory datasetname/Priors", default="no")#"priors_gold_standard.txt")

        parser.add_argument("-pt", "--prior_type", type=str, help="Type of Weights: binary_all (e.g. gold standard prior), real_all (e.g. steady state prior) values", default="no")#"binary_all")

        parser.add_argument("-snf", "--bias_score_splitnodeFeature", type=str, help="Whether to use the prior weights to bias the score of the split node features candidates", required=False, default="")

        parser.add_argument("-thres", "--thres_coeff_var", type=float, help="The coefficient of variation cutoff", default=0)

        parser.add_argument("-timestep", "--time_step", type=str, help="If set to True Time-step method is used otherwise ODE", default="True")

        parser.add_argument("-hypertun", "--tuning_genebygene_randomized", type=str, help="If set to True Gene By Gene Hyper-Params Tuning (Randomized Search) is done", default="False")

        parser.add_argument("-numiter", "--n_iter_search", type=int, help="The number of Randomized Hyper-Params Optmization Iterations", default=10)

        parser.add_argument("-tfa", "--tfa_bool", type=str, help="If set to True TFA is computed and used", default="False")

        parser.add_argument("-nthreads", "--num_threads", type=int, help="Number of threads, it has to match the number of cores allocated. (0 means NO Parallelization) ", default=28)

        parser.add_argument("-savemodels", "--save_models", type=str, help="Save models when running on HPC prince", default="False")

        parser.add_argument("-flagprint", "--flag_print", type=str, help="If this flag True more Output is printed", default="False")

        parser.add_argument("-parse4dyng3", "--parse_4dyng3", type=str, help="If this flag True the dataset will also be parsed to dynGenie3 format and the files printed in the input dir", default="False")

        parser.add_argument("-autometh", "--auto_meth", type=str, help="If this flag is True auto optimization between time-step and ode-log is done", default="True")

        parser.add_argument("-seed", "--rnd_seed", type=int, help="The seed number generator for numpy", default=42)

        args = parser.parse_args()

        #print(args.slurm_job, args.tree_method, "max_feat", args.K, "ntrees", args.ntrees, args.timehorizon, args.datasetname, args.test_size, args.prior_file, args.prior_type, args.data_type, args.data_type_lo, args.bias_score_splitnodeFeature, args.thres_coeff_var, args.num_ets_lo, args.time_step, args.tuning_genebygene_randomized, args.n_iter_search, args.tfa_bool, args.num_threads, args.save_models)

        #Parameters set in pipeline.py/organism.py file will overwrite the args passed by commandline
        #Some params which are not allowed to be set via config file can be safely set as args via commandline
        self.call_run_obj("RF", args.slurm_job, args.tree_method, args.K, args.ntrees, args.timehorizon, args.datasetname, args.test_size, args.prior_file, args.prior_type, args.data_type, args.data_type_lo, args.bias_score_splitnodeFeature, args.thres_coeff_var, args.num_ets_lo, args.time_step, args.tuning_genebygene_randomized, args.n_iter_search, args.tfa_bool, args.num_threads, args.save_models, args.flag_print, args.parse_4dyng3, args.auto_meth, args.rnd_seed)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def str_to_bool(self, s):
        if s == "True":
            return True
        elif s == "False":
            return False
        else:
            raise ValueError