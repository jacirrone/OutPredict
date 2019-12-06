"""
RF_model Class Implementation

"""



import numpy as np
import pandas as pd
import time
from sklearn.tree.tree import BaseDecisionTree
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn import ensemble
from operator import itemgetter
import os
import pwd
import pickle
from multiprocessing import Pool
#from copyreg import pickle
from types import MethodType
import gc
import scipy as sp
#from guppy import hpy
#from memory_profiler import profile
#from joblib import *
from math import exp

def rf_comb_single_unpack(*arg, **kwarg):
    return RF_model.rf_comb_single(arg[0][0], *arg[0][1])


class RF_model():



    def __init__(self, datasetname, data_type, data_type_lo, path_dataset, genelist, tflist, X_train, prior_file, prior_type, y_train, num_features, goldstandard, tuning_genebygene_randomized, X_test_ss, X_test_ts, x_test_ts_current_timepoint, deltas, tau, time_step, save_models, flag_print, rnd_seed):

        self.data_type = data_type
        self.data_type_lo = data_type_lo

        self.datasetname = datasetname
        self.path_dataset = path_dataset
        # self.datatype = datatype
        self.genelist = genelist
        self.tflist = tflist
        self.numgenes = len(genelist)
        self.numtfs = len(tflist) #numOffeatures is different from num of tfs with timeseries
        self.X_train = X_train
        self.y_train = y_train
        self.X_test_ss = X_test_ss
        self.X_test_ts = X_test_ts
        #self.X_test = X_test
        # self.y_test = y_test
        self.prior_file = prior_file
        self.prior_type = prior_type
        # self.num_lo_points = num_lo_points
        self.num_train_points = y_train.shape[0]#self.num_train_points = y_train.shape[1]

        self.num_features = num_features

        self.goldstandard = goldstandard

        self.tuning_genebygene_randomized = tuning_genebygene_randomized

        self.x_test_ts_current_timepoint = x_test_ts_current_timepoint
        self.deltas = deltas
        self.tau = tau

        self.time_step = time_step

        self.feature_weights = np.ones((int(self.numgenes), self.num_features))

        self.save_models = save_models

        self.flag_print = flag_print

        np.random.seed(rnd_seed)
        self.rnd_seed = rnd_seed

        # import sys, site
        # # sys.path.remove(site.USER_SITE)
        # psys_arr = []
        # for psys in sys.path:
        #     if psys.__contains__(site.USER_SITE):
        #         psys_arr.append(psys)
        # for psys_to_remove in psys_arr:
        #     sys.path.remove(psys_to_remove)
        # print('\n'.join(sys.path))


    def compute_feature_importances(self, estimator, tree_method):
        if isinstance(estimator, BaseDecisionTree):
            return estimator.tree_.compute_feature_importances(normalize=False)
        else:
            #this(normalize=False) is to avoid the internal normalization "e.feature_importances_"
            #that makes the features_importances sum up to 1
            #http://stackoverflow.com/questions/15810339/how-are-feature-importances-in-randomforestclassifier-determined
            #feature_importances are computed as described in [Breiman et al. 1984]
            #http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py

            if tree_method == "RF-mod":
                importances = [e.tree_.compute_feature_importances(normalize=False)#[0]
                    for e in estimator.estimators_]
            elif tree_method == "RF":
                importances = [e.tree_.compute_feature_importances(normalize=False)
                    for e in estimator.estimators_]
            elif tree_method == "GB":
            #"[0]" (at the end of compute_feature_importances(normalize=False))
            #because at one point I modified sklearn and I was returning from compute_feature_importances
            #two vectors and one was the features importances
                importances = [e[0].tree_.compute_feature_importances(normalize=False)
                    for e in estimator.estimators_]

            importances = np.asarray(importances)
            return np.sum(importances,axis=0) / len(estimator)

    def plotBar(self, yvalues, pdfname, nfeatures, yname, output_path):
            #from matplotlib.backends.backend_pdf import PdfPages
            #pp = PdfPages(pdfname+'_BarPlotfeatures'+yname+'.pdf')
            plt.clf()
            plt.close()
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(1,1,1)

            ax.bar(list(range(0,nfeatures)), yvalues)

            ax.set_title("BarPlot of features\' "+yname)
            if nfeatures>40:
                ax.set_xticks(np.arange(0,nfeatures, 2))
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(5)

            ax.set_xlabel('Features')
            ax.set_ylabel(yname)

            if self.flag_print:
                plt.savefig(output_path+"/"+pdfname+'_BarPlotfeatures'+yname+'.pdf')
            #pp.savefig(fig)
            #pp.close()

    #@profile
    def rf_comb_single(self, output_idx, tree_method, num_of_feats, ntrees, output_path, output_path_estimators, default_username, n_iter_search_par):
    #def rf_comb_single(self, output_idx, tree_method, num_of_feats, ntrees, h, feature_weights, output_path):
        treeEstimator = None
        gc.collect()
        np.random.seed(self.rnd_seed)
        #Debug
        #print output_idx, tree_method, num_of_feats, ntrees, feature_weights, output_path, output_path_estimators, default_username, n_iter_search_par

        genename = self.genelist[output_idx]

        namemodel = genename+"_Model"

        #self.plotBar(feature_weights, namemodel, len(feature_weights), "Weights", output_path)

        # #Debug
        # print "true or false ", self.ds_instance.index_tfs[output_idx] == True
        # print "len feature_weights ", len(feature_weights)
        # print "ninputs ", ninputs
        # print "input_matrix_time_1_n.shape ", input_comb.shape

        # Parameters of the tree-based method
        if tree_method == 'RF-mod':
            print_method = 'Random Forests Modified'
            if num_of_feats == 'sqrt':
                treeEstimator = RandomForestRegressor(n_estimators=ntrees,max_features="sqrt", n_jobs=-1, oob_score=True, random_state=self.rnd_seed)#, name_model = namemodel, outputpath = output_path)
            elif num_of_feats == 'all':
                treeEstimator = RandomForestRegressor(n_estimators=ntrees,max_features="auto", n_jobs=-1, oob_score=True, random_state=self.rnd_seed)#, name_model = namemodel, outputpath = output_path)
            else:
                if num_of_feats < self.num_features:
                    treeEstimator = RandomForestRegressor(n_estimators=ntrees,max_features=num_of_feats, n_jobs=-1, oob_score=True, random_state=self.rnd_seed)#, name_model = namemodel, outputpath = output_path)
                else:
                    treeEstimator = RandomForestRegressor(n_estimators=ntrees,max_features="auto", n_jobs=-1, oob_score=True, random_state=self.rnd_seed)#, name_model = namemodel, outputpath = output_path)
        elif tree_method == 'RF':
            print_method = 'Random Forests'
            if num_of_feats == 'sqrt':
                treeEstimator = RandomForestRegressor(n_estimators=ntrees, max_features="sqrt", n_jobs=-1, oob_score=True, random_state=self.rnd_seed)#treeEstimator = RandomForestRegressor(n_estimators=ntrees,max_features="sqrt", n_jobs=-1, oob_score=True)
            elif num_of_feats == 'all':
                treeEstimator = RandomForestRegressor(n_estimators=ntrees,max_features="auto", n_jobs=-1, oob_score=True, random_state=self.rnd_seed)
            else:
                if num_of_feats < ninputs:
                    treeEstimator = RandomForestRegressor(n_estimators=ntrees,max_features=num_of_feats, n_jobs=-1, oob_score=True, random_state=self.rnd_seed)
                else:
                    treeEstimator = RandomForestRegressor(n_estimators=ntrees,max_features="auto", n_jobs=-1, oob_score=True, random_state=self.rnd_seed)
            #print "Tree method = %s, num_of_feats = %d, %d trees, time lag = %d" % (print_method,print_K,ntrees,h)
        elif tree_method == 'GB':
            print_method = 'GradientBoostingRegressor'
            if num_of_feats == 'sqrt':
                treeEstimator = ensemble.GradientBoostingRegressor(max_features="sqrt", n_estimators=ntrees, random_state=self.rnd_seed)#, max_depth=3, loss="lad")#"huber")
            elif num_of_feats == 'all':
                treeEstimator = ensemble.GradientBoostingRegressor(max_features="auto", n_estimators=ntrees, random_state=self.rnd_seed)#(n_estimators=ntrees,max_features="auto", n_jobs=-1)
            else:
                if num_of_feats < self.num_features:
                    treeEstimator = ensemble.GradientBoostingRegressor(n_estimators=ntrees,max_features=num_of_feats, random_state=self.rnd_seed)
                else:
                    treeEstimator = ensemble.GradientBoostingRegressor(n_estimators=ntrees,max_features="auto", random_state=self.rnd_seed)

        n_iter_search = n_iter_search_par

        if self.tuning_genebygene_randomized and tree_method == 'RF':
            #from sklearn.grid_search import GridSearchCV
            from sklearn.model_selection import RandomizedSearchCV
            from scipy.stats import randint as sp_randint
            np.random.seed(self.rnd_seed)
            # specify parameters and distributions to sample from
            param_dist = {"n_estimators": [ntrees],
                    "min_samples_split": sp_randint(2, 30),
                "max_depth" : [1, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30],
                "min_samples_leaf" : [1, 2, 4, 6, 8, 10, 20, 30, 40, 50, 65, 80],
            "oob_score": [True],
            "max_features": ["sqrt"]}

            # run randomized search
            random_search = RandomizedSearchCV(treeEstimator, param_distributions=param_dist,
                                       n_iter=n_iter_search, n_jobs=-1)#cv=2
            #"max_depth"         : [2, 5, 10, 20, 25],
                #"min_samples_split" : [2, 4, 6, 8, 10, 15],

            #"max_depth" : [1, 5, 10, 15, 20, 25, 30],
                #"min_samples_leaf" : [1, 2, 4, 6, 8, 10]}
            random_search.fit(self.X_train, self.y_train[:, output_idx])
            #print grid_search.best_params_

            #X [n_samples, n_features]
            #y [n_samples, num_genes]
            #print random_search.best_params_
            treeEstimator = RandomForestRegressor(**random_search.best_params_)

        elif self.tuning_genebygene_randomized and tree_method == 'GB':
            #from sklearn.grid_search import GridSearchCV
            from sklearn.model_selection import RandomizedSearchCV
            from scipy.stats import randint as sp_randint
            np.random.seed(self.rnd_seed)
            # specify parameters and distributions to sample from
            param_dist = {"loss" : ['ls', 'lad', 'huber', 'quantile'],
                    "n_estimators": [10, 50, 100, 170, 260, 350],
                    "max_features": ["sqrt"],
                    "min_samples_split": sp_randint(2, 30),
                    "max_depth" : [1, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30],
                    "min_samples_leaf" : [1, 2, 4, 6, 8, 10, 20, 30, 40, 50, 65, 80]}



            # run randomized search
            random_search = RandomizedSearchCV(treeEstimator, param_distributions=param_dist,
                               n_iter=n_iter_search, n_jobs=-1)#cv=2
            #"max_depth"         : [2, 5, 10, 20, 25],
            #"min_samples_split" : [2, 4, 6, 8, 10, 15],

            #"max_depth" : [1, 5, 10, 15, 20, 25, 30],
            #"min_samples_leaf" : [1, 2, 4, 6, 8, 10]}
            random_search.fit(self.X_train, self.y_train[:, output_idx])
            #print random_search.best_params_
            #X [n_samples, n_features]
            #y [n_samples, num_genes]
            treeEstimator = ensemble.GradientBoostingRegressor(**random_search.best_params_)

        if tree_method == "RF-mod":
            #Debug
            #print self.feature_weights[output_idx,:]
            treeEstimator.fit(self.X_train, self.y_train[:, output_idx], feature_weight = self.feature_weights[output_idx,:])
        else:
            #Debug
            # print output_idx
            # print self.X_train.shape
            # print self.y_train[:, output_idx].shape

            treeEstimator.fit(self.X_train, self.y_train[:, output_idx])

            if default_username == "jc3832" and self.save_models == "True":
                with open(output_path_estimators+'/Gene'+str(output_idx), 'wb') as f:
                    pickle.dump(treeEstimator, f)
                f.close()
            # # in your prediction file
            # with open(output_path_estimators+'/Gene'+str(output_idx), 'rb') as f:
            #     treeEstimator = cPickle.load(f)



        # if tree_method == "RF-mod":
        # 	treeEstimator.fit(self.X_train[output_idx, :], self.y_train[output_idx, :], feature_weight = feature_weights)
        # else:
        # 	treeEstimator.fit(self.X_train[output_idx, :], self.y_train[output_idx, :])


        #Explanatory variables(i.e. matrix_input) VS Response Variable(i.e. output)
        #http://www.statisticshowto.com/explanatory-variable/

        #Debug
        # print "After fit"
        # print "genename", genename
        # print "fweight", feature_weights

        # Compute importance scores - e.tree_.compute_feature_importances(normalize=False) in an my own old version of scikit-learn
        # returned 3 variables from the cython function, [0] is the featureimportance vector.
        feature_importances = self.compute_feature_importances(treeEstimator, tree_method)

        #print "true or false ", self.ds_instance.index_tfs[output_idx] == True
        # print "len feat_importances ", len(feature_importances)
        # print "len feature_weights ", len(feature_weights)
        # print "ninputs ", self.num_features
        # print "input_matrix", self.X_train[output_idx, :].shape

        #self.plotBar(feature_importances, namemodel, len(feature_weights), "Importances", output_path)


        treeEstimator.feature_importances_

        causal_tfs = np.zeros(self.numtfs)#(self.numgenes)

        causal_tfs = feature_importances
        # if self.num_features == self.numtfs + 1: #e.g. timeseries
        # 	causal_tfs[:self.numtfs] = feature_importances[:-1]
        # 	causal_tfs[output_idx] = feature_importances[-1]
        # elif self.num_features == self.numtfs: #e.g. steadystate
        # 	causal_tfs[:self.numtfs] = feature_importances



        causal_tfs = causal_tfs / np.sum(causal_tfs)#hence the sum of feature_importances sum up to one
        #Debug -
        #print "causal_tfs", causal_tfs
        #print "sum(causal_tfs)", sum(causal_tfs)

        #Training error
        y_train_pred_single = treeEstimator.predict(self.X_train)
        #y_train_pred_single = treeEstimator.predict(self.X_train[output_idx, :])

        y_test_pred_single_ss = ""
        y_test_pred_single_ts = ""
        y_test_pred_single_future_time_points = ""

        if self.data_type == "SS" or (self.data_type == "TS-SS" and (self.data_type_lo=="TS-SS" or self.data_type_lo=="SS")):
            if self.X_test_ss.shape[0] > 0:
                y_test_pred_single_ss = treeEstimator.predict(self.X_test_ss)
            else:
                y_test_pred_single_ss = 0.0000000000001

        if self.data_type == "TS" or (self.data_type == "TS-SS" and (self.data_type_lo=="TS-SS" or self.data_type_lo=="TS")):
            if self.X_test_ts.shape[0] > 0:
                y_test_pred_single_ts = treeEstimator.predict(self.X_test_ts)
                if self.time_step:
                    y_test_pred_single_future_time_points = y_test_pred_single_ts
                else:
                    tau_vect = np.asarray([float(self.tau)] * len(self.deltas))
                    y_test_pred_single_future_time_points = (self.deltas / tau_vect) *  (y_test_pred_single_ts - self.x_test_ts_current_timepoint[:, output_idx])  + self.x_test_ts_current_timepoint[:, output_idx]
                    #alpha way
                    #y_test_pred_single_future_time_points = (self.deltas * y_test_pred_single_ts) - (tau_vect * self.deltas * self.x_test_ts_current_timepoint[:,output_idx]) + self.x_test_ts_current_timepoint[:, output_idx]

                    #interp_res = (float(tau) / float(following_delt[cntr])) * (exp_mat[cond[j]].astype('float64') - exp_mat[cond[i]].astype('float64')) + exp_mat[cond[i]].astype('float64')
                    #y_test_pred_single_future_time_points = (self.deltas * y_test_pred_single_ts) - ((self.x_test_ts_current_timepoint[:, output_idx]) * (np.asarray(self.deltas) / self.tau)) + self.x_test_ts_current_timepoint[:, output_idx]
            else:
                y_test_pred_single_ts = 0.0000000000001
                y_test_pred_single_future_time_points = 0.0000000000001

            #y_test_pred_single = treeEstimator.predict(self.X_test[output_idx, :])


        # if tree_method == "RF-mod":
        # 	y_hat_stdDev_trees = np.std(treeEstimator.all_y_hat, axis=0) #Std Deviation computed across ntrees
        # else:
        # 	y_hat_stdDev_trees = 0


        if tree_method == "RF-mod" or tree_method == "RF":
            oobpredictions = treeEstimator.oob_prediction_
            oobscore_r2 = treeEstimator.oob_score_  # which is the MSE of
        # treeEstimator.oob_prediction_ and y_train; oob_prediction_ has the same length of
        # y_train, and for each index contains the avg of the predictions for the trees
        # for which that data point wasn't sampled.(some of the indexes are 0).
        if tree_method == "GB":
            oobpredictions = 0
            oobscore_r2 = 0  # treeEstimator.oob_improvement_


        gc.collect()
        #return causal_tfs, treeEstimator, y_train_pred_single, y_test_pred_single, y_hat_stdDev_trees
        #return causal_tfs, treeEstimator, y_train_pred_single, y_test_pred_single_ss, y_test_pred_single_ts, y_test_pred_single_future_time_points

        return causal_tfs, oobpredictions, oobscore_r2, y_train_pred_single, y_test_pred_single_ss, y_test_pred_single_ts, y_test_pred_single_future_time_points

    def get_feature_weights(self, output_path):

        targets_tfs_weights_df = None

        if self.prior_type == "real_all":
            #Format: list of edges with weights
            try:
                if self.flag_print:
                    #print(self.path_dataset+"priors/"+self.prior_file)
                    print(self.path_dataset+self.prior_file)
                prior_file_df = pd.read_table(self.path_dataset+self.prior_file, header=None)
            except:
                # raise ValueError('Error: Either Priors file doesnt exist or Check Prior file ("steady_state"/"Real Num") is in the right format')
                print('Error: Either Priors file doesnt exist or Check Prior file ("steady_state"/"Real Num") is in the right format')
                exit(1)


            prior_file_df.columns = ['source','dest','weight']

            prior_file_df.set_index(['source','dest'], inplace=True)

            prior_file_df.index.names = ['TFs','gene']

            targets_tfs_weights_df = pd.DataFrame()

            #selfweight, truepos =  [2, 2]#2] #[mt.exp(1),mt.exp(1)]

            #trueneg = 1 #float(1)/float(truepos)

            targets_priors = prior_file_df.index.get_level_values('gene').unique()
            tfs_priors = prior_file_df.index.get_level_values('TFs').unique()

            if self.flag_print:
                print("Num of Target genes in the priors: ", len(targets_priors))
                print("Num of TFs genes in the priors: ", len(tfs_priors))

                print("Num of genes targets intersecting priors and universe is: ", len(set.intersection(set(targets_priors), set(self.genelist))))
                print("Num of TFs intersecting priors and universe is: ", len(set.intersection(set(tfs_priors), set(self.tflist))))

            #Debug
            #print self.tflist
            count=0
            for g in self.genelist:
                #try:
                if self.num_features == len(self.tflist):
                    tfs_weights_tmp = pd.DataFrame({'TFs': self.tflist, 'weight': 1})
                elif self.num_features == len(self.tflist) + 1:
                    tfs_weights_tmp = pd.DataFrame({'TFs': np.append(self.tflist,g), 'weight': 1})

                #Vector of Weights of TFs for a given gene g
                TFs_g_fc_tmp = prior_file_df[prior_file_df.index.get_level_values('gene')==g]
                #print TFs_g_fc_tmp
                #Debug

                tfs_weights_tmp.set_index('TFs', inplace=True)
                for ind_tf, tf in enumerate(self.tflist):
                    if tf!=g:
                        try:
                            weight = abs(TFs_g_fc_tmp.loc[tf,g]['weight'])
                            tfs_weights_tmp.loc[tf] = weight + 1
                            if weight > 1:
                                raise ValueError('Steady state priors is based on weights less than 1, change the algorithm')
                            # if weight < 1 and weight > 0.09999:
                            #     tfs_weights_tmp.loc[tf] = 2 + weight
                            # elif weight < 0.1:
                            #     tfs_weights_tmp.loc[tf] = weight + 0.5
                            # else:
                            #     raise ValueError('weight less than zero')
                        except:
                            #print "No prior info for gene: ", g
                            #count = count + 1
                            #print count
                            pass
                    else:
                        #Self-interaction
                        if self.num_features == len(self.tflist):#steady state
                            tfs_weights_tmp.loc[tf] = 0
                        elif self.num_features == len(self.tflist) + 1: #timeseries
                            tfs_weights_tmp.iloc[ind_tf] = 0
                            tfs_weights_tmp.iloc[len(self.tflist)] = np.max(tfs_weights_tmp['weight'])


                #Self-interaction
                #tfs_weights_tmp.loc[g] = np.max(tfs_weights_tmp['weight'])

                targets_tfs_weights_df_tmp = pd.concat([tfs_weights_tmp, pd.DataFrame()], keys=[g,""], sort=True)

                targets_tfs_weights_df_tmp.index.names=["Genes","TFs"]

                targets_tfs_weights_df = pd.concat([targets_tfs_weights_df, targets_tfs_weights_df_tmp], sort=True)
                #except:
                #    raise ValueError('universe genes (i.e. of the expression dataset) must be in the prior')
            if self.flag_print:
                print(count)
        elif self.prior_type == "binary_all":
            #Format: adjacency matrix - all gold standard - take abs value just in case there are -1
            try:
                # print self.path_dataset+"priors/"+self.prior_file
                prior_file_df = pd.read_table(self.path_dataset+"priors/"+self.prior_file)
                if self.flag_print:
                    print(self.path_dataset+"priors/"+self.prior_file)
                    #print(self.path_dataset+self.prior_file)
                # prior_file_df = pd.read_table(self.path_dataset+self.prior_file)
            except:
                raise ValueError('Priors file doesnt exist')
            try:
                prior_file_df.set_index("Unnamed: 0", inplace=True)
                prior_file_df.index.names = [None]
            except:
                if self.flag_print:
                    print("No index Unnamed: 0")
            try:
                prior_file_df.set_index("index", inplace=True)
                prior_file_df.index.names = [None]
            except:
                if self.flag_print:
                    print("No index named index")


            targets_tfs_weights_df = pd.DataFrame()

            selfweight, truepos = [exp(1), exp(1)] #[5, 4] #[mt.exp(1),mt.exp(1)]

            trueneg = 1/exp(1) #1 #float(1)/float(truepos)

            genes_row_gs = list(prior_file_df.index.values)
            tfs_col_gs = list(prior_file_df.columns)

            #Debug
            #print prior_file_df.shape
            #print prior_file_df

            for g in self.genelist:
                #try:
                if self.num_features == len(self.tflist):
                    tfs_weights_tmp = pd.DataFrame({'TFs': self.tflist, 'weight': 1})
                elif self.num_features == len(self.tflist) + 1:
                    tfs_weights_tmp = pd.DataFrame({'TFs': np.append(self.tflist,g), 'weight': 1})

                #print tfs_weights_tmp

                tfs_weights_tmp.set_index('TFs', inplace=True)

                #print str(g in genes_row_gs)
                if g in genes_row_gs:
                    #Vector of Weights of TFs for a given gene g from goldstandard
                    tfs_weight = prior_file_df.loc[g, :]
                    #tfs_col_gs subset of tflist
                    for ind_tf, tf in enumerate(tfs_col_gs):
                        if tf!=g:
                            weight = abs(tfs_weight[ind_tf])
                            if weight == 1:
                                tfs_weights_tmp.loc[tf] = truepos
                            elif weight == 0:
                                tfs_weights_tmp.loc[tf] = trueneg
                            else:
                                raise ValueError('gold standard has to have values either 1 or 0')
                        else:
                            #Self-interaction; Former version was different, now TS and steady state are integrated so num_features = num of tfs
                            if self.num_features == len(self.tflist):#steady state
                                tfs_weights_tmp.loc[tf] = selfweight
                            elif self.num_features == len(self.tflist) + 1: #timeseries
                                #find index of tf in the original tflist
                                indx_orig_tflist = list(self.tflist).index(tf)
                                tfs_weights_tmp.iloc[indx_orig_tflist] = 0

                                tfs_weights_tmp.iloc[len(self.tflist)] = selfweight
                                #print "selfweight", selfweight




                targets_tfs_weights_df_tmp = pd.concat([tfs_weights_tmp, pd.DataFrame()], keys=[g,""], sort=True)

                targets_tfs_weights_df_tmp.index.names=["Genes","TFs"]

                targets_tfs_weights_df = pd.concat([targets_tfs_weights_df, targets_tfs_weights_df_tmp], sort=True)
            #except Exception, e:
            #logger.error('Exception error '+ str(e))
            #raise ValueError('universe genes (i.e. of the expression dataset) must be in the prior')
        else:
            raise ValueError('Wrong Prior Type')

        #targets_tfs_weights_df.to_csv(output_path+"/_weights_df.txt", sep="\t")
        return targets_tfs_weights_df

        #if binary 1 and 0, do e^1 - e^0
        #if real values between 0 and 1 - add 1

    #@profile
    def rf_comb(self, nthreads, output_path, index_steady_state_new, index_time_points_new, design_mat, delta_vect, res_mat2, regulators='all', tree_method='RF-mod', num_of_feats='sqrt', ntrees=1000, n_iter_search=10):
        #def rf_comb(self, output_path, num_ets_lo, regulators='all', tree_method='RF-mod', num_of_feats='sqrt', ntrees=1000, h=1, ):
        # def _pickle_method(method):
        #     func_name = method.im_func.__name__
        #     obj = method.im_self
        #     cls = method.im_class
        #     return _unpickle_method, (func_name, obj, cls)

        # def _unpickle_method(func_name, obj, cls):
        #     for cls in cls.mro():
        #         try:
        #             func = cls.__dict__[func_name]
        #         except KeyError:
        #             pass
        #         else:
        #             break
        #     return func.__get__(obj, cls)

        # import copy_reg
        # import types
        # copy_reg.pickle(types.MethodType, self._pickle_method, self._unpickle_method)
        time_start = time.time()

        output_path_estimators = ""
        getlogin = lambda: pwd.getpwuid(os.getuid())[0]
        default_username = getlogin()
        if default_username=="jc3832" and self.save_models == "True":
            output_path_estimators = "/scratch/jc3832/ExperimentsMay2018/"+output_path.split('/')[-1]+"/Estimators"
            if not os.path.exists(output_path_estimators):
                os.makedirs(output_path_estimators)

        if self.prior_file != "no":
            targets_tfs_weights_df = self.get_feature_weights(output_path)


        causal_matrix = np.zeros((self.numgenes,self.numtfs))

        arr_treeEstimator = []

        y_train_pred = np.zeros((self.numgenes, self.num_train_points))

        y_test_pred_ss = ""
        y_test_pred_ts = ""
        y_test_pred_future_time_points = ""

        if self.data_type == "SS" or (self.data_type == "TS-SS" and (self.data_type_lo=="TS-SS" or self.data_type_lo=="SS")):
            y_test_pred_ss= np.zeros((self.numgenes, self.X_test_ss.shape[0]))

        if self.data_type == "TS" or (self.data_type == "TS-SS" and (self.data_type_lo=="TS-SS" or self.data_type_lo=="TS")):
            y_test_pred_ts= np.zeros((self.numgenes, self.X_test_ts.shape[0]))
            y_test_pred_future_time_points = np.zeros((self.numgenes, self.x_test_ts_current_timepoint.shape[0]))
        #y_test_pred= np.zeros((self.numgenes, self.num_lo_points))

        # y_hat_stdDev_trees_arr = np.zeros((self.numgenes, self.num_lo_points))

        # y_act_pred_corr = np.zeros((self.numgenes, 2))

        # y_act_pred_corr2 = np.zeros((self.numgenes, 2))

        # estimators = []

        df = pd.DataFrame(0, index = self.genelist, columns = ['oobscore', 'MSE_oob', 'MSE_test'])
        df = df.astype(float)

        if self.prior_file != "no":
            #to debug: total num of positive edges
            tot_truepos = 0
            for j in range(int(self.numgenes)):
                genename = self.genelist[j]
                self.feature_weights[j, :] = targets_tfs_weights_df.loc[genename, :]["weight"].values
                tot_truepos = tot_truepos + (np.sum(self.feature_weights[j, :]==exp(1)))

            if self.flag_print:
                print("Total number of true positive edges", tot_truepos)

        #h = hpy()
        #print "RF_model"+str(h.heap()) + "\n"
        # outfile.write("RF_model"+str(h.heap()) + "\n")

        time_end = time.time()
        if self.flag_print:
            print("Elapsed time for parsing priors when OP is used with priors: %.2f seconds" % (time_end - time_start))
            df.to_csv(output_path+"/_output_predictions_results.txt", sep="\t")

        time_start = time.time()

        if nthreads==0:

            alloutput = []
            for i in range(int(self.numgenes)):

                output_tmp = self.rf_comb_single(i, tree_method, num_of_feats, ntrees, output_path, output_path_estimators, default_username, n_iter_search)
                alloutput.append(output_tmp)

        else:

            # OLD causal_tfs, treeEstimator, y_train_pred_single, y_test_pred_single, y_hat_stdDev_trees = self.rf_comb_single(i, tree_method, num_of_feats, ntrees, h, feature_weights, output_path)

            #causal_tfs, treeEstimator, y_train_pred_single, y_test_pred_single_ss, y_test_pred_single_ts, y_test_pred_single_future_time_points = self.rf_comb_single(i, tree_method, num_of_feats, ntrees, feature_weights, output_path, output_path_estimators, default_username, n_iter_search)
            input_data = [[i, tree_method, num_of_feats, ntrees, output_path, output_path_estimators, default_username, n_iter_search] for i in range(int(self.numgenes))]

            # pool = Pool(nthreads)

            # alloutput = pool.map(self.rf_comb_single, input_data)

            # with poolcontext(processes=nthreads) as pool:
            # 	alloutput = pool.map(rf_comb_single_unpack, input_data)

            #Debug
            # print len(input_data)
            # print input_data[1]


            # import distributed
            # from joblib import DistributedBackend
            # register_parallel_backend('distributed', DistributedBackend, make_default=True)

            # from sklearn.joblib import *
            #
            #
            # from sklearn.externals import joblib
            # #import distributed.joblib  # register the dask joblib backend
            #
            # from dask.distributed import Client
            # client = Client()
            #
            # # est = ParallelEstimator()
            # # gs = GridSearchCV(est)
            #
            # with joblib.parallel_backend('dask'):


            #pool = Pool(processes=nthreads)
            import concurrent.futures
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=nthreads)

            alloutput = pool.map(rf_comb_single_unpack,list(zip([self]*len(input_data), input_data)))


        for i, out in enumerate(alloutput):

            causal_tfs='';

            treeEstimator='';

            y_train_pred_single='';

            y_test_pred_single_ss='';

            y_test_pred_single_ts='';

            y_test_pred_single_future_time_points = ''
            #y_test_pred_single='';

            # y_hat_stdDev_trees='';

            #Debug
            # print "out", out
            # print "out 0", out[0]
            # print "out 1", out[1]

            (causal_tfs, oobpredictions, oobscore_r2, y_train_pred_single, y_test_pred_single_ss, y_test_pred_single_ts, y_test_pred_single_future_time_points) = out


            # y_test_single = self.y_test[i, :]

            # mse_test = mean_squared_error(y_test_single, y_test_pred_single)


            # y_act_pred_corr[i, :] = scipy.stats.pearsonr(y_test_single, y_test_pred_single)


            # if num_ets_lo>1:
            # 	num_timepoints = len(y_test_single) / num_ets_lo

            # 	corr_vect = []
            # 	pvalue_vect = []

            # 	for nts in range(0, num_ets_lo):

            # 		corr_pvale_tmp = scipy.stats.pearsonr(y_test_single[nts*num_timepoints: (num_timepoints + (nts*num_timepoints))] , y_test_pred_single[nts*num_timepoints: (num_timepoints + (nts*num_timepoints))])

            # 		corr_vect.append(corr_pvale_tmp[0])

            # 		pvalue_vect.append(corr_pvale_tmp[1])


            # 	y_act_pred_corr2[i, 0] = np.mean(corr_vect)

            # 	y_act_pred_corr2[i, 1] = np.std(pvalue_vect)


            # estimators.append(treeEstimator)


            y_train_single = self.y_train[:, i]#y_train_single = self.y_train[i, :]


            # if tree_method == "RF-mod" or tree_method=="RF":
            #     oobpredictions = treeEstimator.oob_prediction_
            #     oobscore_r2 = treeEstimator.oob_score_ #which is the MSE of
            # #treeEstimator.oob_prediction_ and y_train; oob_prediction_ has the same length of
            # #y_train, and for each index contains the avg of the predictions for the trees
            # #for which that data point wasn't sampled.(some of the indexes are 0).
            # if tree_method == "GB":
            #     oobpredictions = 0
            #     oobscore_r2 = 0 #treeEstimator.oob_improvement_

            #TODO: compute naive_sts_pred_comb which is is naive prediction for all training data.

            # if self.datatype == "TS":
            # 	oobpredictions = treeEstimator.oob_prediction_

            # 	indx_nonzero = np.nonzero(oobpredictions)[0]
            # 	abs_err_naive = abs(y_train_single[indx_nonzero] - naive_sts_pred_comb[indx_nonzero])
            # 	abs_err_naive = abs_err_naive.astype(np.float)
            # 	mae_naive = float(np.sum(abs_err_naive))/float(len(indx_nonzero))

            # 	mase = 0

            # 	for unsampled_ind, pred in enumerate(oobpredictions):
            # 		if pred != 0:
            # 			abs_err_pred = abs(output_comb[unsampled_ind] - pred)
            # 			mase += float(abs_err_pred)/mae_naive

            # 	mase = mase/float(len(indx_nonzero))
            # 	oobscore_tmp = mase

            # 	print "mase, ", mase
            # #TODO: Find a criterium for Steady State for rejection similar to Timeseries

            #Since we use oobscore for refusal threshold, we Normalize by the variance in order
            #not to reject the genes with highest variance.

            indx_nonzero = np.nonzero(oobpredictions)[0]
            var_oob_points = float(np.var(y_train_single[indx_nonzero]))

            if len(index_time_points_new) > 0 and self.time_step == False:
                index_time_points_oob = set(index_time_points_new).intersection(set(indx_nonzero))
                index_time_points_oob = np.asarray(list(index_time_points_oob))

                oobpredictions_ts = oobpredictions[index_time_points_oob]

                y_train_single_curr = design_mat.iloc[i, index_time_points_oob].values

                deltas_oob = delta_vect.iloc[:,index_time_points_oob].values.astype('float64')

                tau_vect = np.asarray([float(self.tau)] * len(deltas_oob))

                y_oob_pred_single_future_time_points = (deltas_oob / tau_vect) * (oobpredictions_ts - y_train_single_curr) + y_train_single_curr

                y_oob_actual_single_future_time_points = res_mat2.iloc[i, index_time_points_oob].values

                oob_mse_ts = mean_squared_error(y_oob_actual_single_future_time_points, y_oob_pred_single_future_time_points[0,:])

                oob_mse = oob_mse_ts

                #print "OOB MSE for time series is ", oob_mse_ts

            if len(index_steady_state_new) > 0 and self.time_step == False:
                index_steady_state_oob = set(index_steady_state_new).intersection(set(indx_nonzero))
                index_steady_state_oob = np.asarray(list(index_steady_state_oob))
                oob_mse_ss = mean_squared_error(oobpredictions[index_steady_state_oob], y_train_single[index_steady_state_oob])

                #print "OOB MSE for Steady State is ", oob_mse_ss

                if self.data_type == "SS" or (self.data_type == "TS-SS" and (self.data_type_lo == "TS-SS" or self.data_type_lo == "SS")):
                    if self.data_type_lo == "TS-SS":
                        #if len(index_time_points_oob) > 0:
                        #pass
                        oob_mse = np.mean([oob_mse, oob_mse_ss])
                    elif self.data_type_lo == "SS":
                        oob_mse = oob_mse_ss

            if self.time_step == True:
                oob_mse = mean_squared_error(oobpredictions[indx_nonzero], y_train_single[indx_nonzero])


            #oob_mse_norm_byvar can be used to reject gene model based on CV score.
            if var_oob_points != 0:
                oob_mse_norm_byvar = float(oob_mse) / var_oob_points
            else:
                oob_mse_norm_byvar = 1.5
                # Debug - fed
                if self.flag_print:
                    print("gene", i)
                    print("var_oob_points", var_oob_points)
                    print("oob_mse", oob_mse)
                    print("oobpredictions", oobpredictions)
                indx_nonzero = np.nonzero(oobpredictions)[0]
                if self.flag_print:
                    print("indx_nonzero", indx_nonzero)
                # var_oob_points = float(np.var(y_train_single[indx_nonzero]))

                #Uncomment this when you are not running Ken's
                #raise ValueError("variance oob points equal to 0")


            df.iloc[i, 0] = oobscore_r2

            df.iloc[i, 1] = oob_mse

            df.iloc[i, 2] = 1#mse_test

            causal_matrix[i,:] = causal_tfs

            # y_hat_stdDev_trees_arr[i, :]= y_hat_stdDev_trees

            y_train_pred[i,:] = y_train_pred_single

            #Debug
            #print "oobscore_tmp: ", df.iloc[i, 0]

            if self.data_type == "SS" or (self.data_type == "TS-SS" and (self.data_type_lo=="TS-SS" or self.data_type_lo=="SS")):
                y_test_pred_ss[i,:] = y_test_pred_single_ss

            if self.data_type == "TS" or (self.data_type == "TS-SS" and (self.data_type_lo=="TS-SS" or self.data_type_lo=="TS")):
                y_test_pred_ts[i,:] = y_test_pred_single_ts

                y_test_pred_future_time_points[i, :] = y_test_pred_single_future_time_points
                #y_test_pred[i,:] = y_test_pred_single


        # if num_ets_lo>0:
        # 	num_timepoints = (np.shape(y_test_pred)[1]) / num_ets_lo
        # 	for nts in range(0, num_ets_lo):

        # 		for timepoint in range(1, num_timepoints):

        # 			for gid in range(0, self.numgenes):

        # 				indx_timepoint = (timepoint-1) + nts*num_timepoints

        # 				xtest = np.append(y_test_pred[:self.numtfs, indx_timepoint], y_test_pred[gid, indx_timepoint])

        # 				y_test_pred[gid, indx_timepoint+1] = estimators[gid].predict(xtest.reshape(1, -1))



        #causal_matrix = transpose(causal_matrix)


        time_end = time.time()
        if self.flag_print:
            print("Elapsed time for training the RF models for all genes: %.2f seconds" % (time_end - time_start))
            df.to_csv(output_path+"/_output_predictions_results.txt", sep="\t")
        #
        #h = hpy()
        #print "RF_model"+str(h.heap()) + "\n"
        # outfile.write("RF_model" + str(h.heap()) + "\n")

        return causal_matrix, y_train_pred, y_test_pred_ss, y_test_pred_ts, y_test_pred_future_time_points, np.mean(df.iloc[:, 0]), np.mean(df.iloc[:, 1])
        #return causal_matrix, y_test_pred, y_train_pred, y_hat_stdDev_trees_arr, y_act_pred_corr, y_act_pred_corr2













