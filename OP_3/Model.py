"""
Model Class Implementation

"""



#Example of Ordered Dictionary for timeseries
#dict_X_ts (key, value) -> (genename, matrix of dimension: num_samples X num_of_features)
#dict_y_ts (key, value) -> (genename, vector of dimension: num_samples)

import time
import numpy as np
from RF_model import RF_model
from sklearn.metrics import *
import gc
import pandas as pd
from network_inference import NetworkInference
import scipy
import matplotlib.pyplot as plt
#from guppy import hpy
#from memory_profiler import profile

class Model:

	datatype = ""
	dict_X_train = ""
	dict_y_train = ""
	dict_X_test = ""
	dict_y_test = ""

	def __init__(self, data_type, data_type_lo, genelist, tflist, prior_file, prior_type, script_dir, path_dataset, bias_score_splitnodeFeature, X_test_ss, X_test_ts, y_test_ss, y_test_ts, x_test_ts_current_timepoint, y_test_ts_future_timepoint, deltas, tau, x_test_ts_timepoint0, flag_print, poot, rnd_seed):
		#print("This is the class Model")
		self.data_type = data_type
		self.data_type_lo = data_type_lo
		self.genelist = genelist
		self.tflist = tflist
		self.numgenes = len(genelist)
		self.numtfs = len(tflist)
		self.prior_file = prior_file
		self.prior_type = prior_type
		self.script_dir = script_dir
		self.path_dataset = path_dataset
		self.bias_score_splitnodeFeature = bias_score_splitnodeFeature
		self.X_test_ss = X_test_ss
		self.X_test_ts = X_test_ts
		self.y_test_ss = y_test_ss
		self.y_test_ts = y_test_ts
		self.x_test_ts_current_timepoint = x_test_ts_current_timepoint
		self.y_test_ts_future_timepoint = y_test_ts_future_timepoint
		self.deltas = deltas
		self.tau = tau
		self.x_test_ts_timepoint0 = x_test_ts_timepoint0
		self.flag_print = flag_print
		self.poot = poot
		# import sys, site
		# # sys.path.remove(site.USER_SITE)
		# psys_arr = []
		# for psys in sys.path:
		# 	if psys.__contains__(site.USER_SITE):
		# 		psys_arr.append(psys)
		# for psys_to_remove in psys_arr:
		# 	sys.path.remove(psys_to_remove)
		# print('\n'.join(sys.path))

		np.random.seed(rnd_seed)
		self.rnd_seed = rnd_seed

	def get_stsXtestvalues_4timeseries(self, dict_X_ts_lo):
		x_test = dict_X_ts_lo[list(dict_X_ts_lo.keys())[1]]
		y_test_stsNaive = np.zeros((self.numgenes, x_test.shape[0]))


		for ind, key in enumerate(dict_X_ts_lo.keys()):
			x_test = dict_X_ts_lo[key]
			y_test_stsNaive[ind,:] = x_test[:, -1]

		return y_test_stsNaive

	def fromdict_toarr(self, dict_data):
		value = dict_data[list(dict_data.keys())[1]]
		if len(value.shape) == 1:
			matrix_ = np.zeros((self.numgenes, value.shape[0]))
		elif len(value.shape) == 2:
			matrix_ = np.zeros((self.numgenes, value.shape[0], value.shape[1]))
		elif len(value.shape) == 3:
			matrix_ = np.zeros((self.numgenes, value.shape[0], value.shape[1], value.shape[2]))

		for ind, key in enumerate(dict_data.keys()):
			value = dict_data[key]
			matrix_[ind,:] = value

		return matrix_

	#Function from old version in Model.py
	# def loadTrainingTestSet(self, datatype, dict_X_train_ts, dict_y_train_ts, dict_X_test_ts, dict_y_test_ts, dict_X_train_ss, dict_y_train_ss, dict_X_test_ss, dict_y_test_ss):
	# 	self.datatype = datatype
	#
	# 	if datatype == "TS":
	# 		#Feature vector has one more dimension (since the target gene is included to the features)
	# 		self.dict_X_train = dict_X_train_ts
	# 		self.dict_y_train = dict_y_train_ts
	# 		self.dict_X_test = dict_X_test_ts
	# 		self.dict_y_test = dict_y_test_ts
	# 	elif datatype == "SS":
	# 		self.dict_X_train = dict_X_train_ss
	# 		self.dict_y_train = dict_y_train_ss
	# 		self.dict_X_test = dict_X_test_ss
	# 		self.dict_y_test = dict_y_test_ss
	# 	elif datatype == "TS-SS":
	# 		print("Still to implement: Add a feature at the end of the feature vector of the steady state data and assign value 0")

	def get_mse(self, yname, ngenes, y, y_pred, outputpath=""):
		std_mse = 0
		if y.shape[1] == 0:
			mse = 0
		else:
			mse_arr = np.zeros(ngenes)

			n_actual_genes = ngenes

			if ngenes>1:
				mse = 0
				for i in range(0,ngenes):
					#print "Gene: ", i
					actual_y = np.copy(np.asarray(y[i,:]))
					pred_y = np.copy(np.asarray(y_pred[i,:]))


					#Don't correlate conditions where a given gene has value 0s
					where_zeros = np.where(np.asarray(actual_y)==0)[0]
					if where_zeros.size > 0:
						actual_y = np.asarray(np.delete(np.asarray(actual_y), where_zeros))
						pred_y =  np.asarray(np.delete(np.asarray(pred_y), where_zeros))

					if len(actual_y) !=0 and len(pred_y) != 0:
						tmp_mse = mean_squared_error(actual_y, pred_y)
					else:
						tmp_mse = 0
					mse += tmp_mse
					mse_arr[i] = tmp_mse
					if tmp_mse == 0:
						n_actual_genes -= 1
			else:
				mse = mean_squared_error(y, y_pred)
				mse_arr = mse

			if ngenes>1:
				#print "the number of genes with good predictions are: ", n_actual_genes
				mse = float(mse) / float(n_actual_genes)
				std_mse = np.std(mse_arr[mse_arr!=0])
			if self.flag_print:
				print("Inside get mse function MSE AVG for var name "+yname+" across genes= " + str(mse)+"\n")

		#Plot MSE for each gene
		#     plt.clf()
		#     plt.close()
		#     fig = plt.figure()
		#     ax1 = fig.add_subplot(111)
		#     ax1.plot(mse_arr, marker='o')
		#     #ax1.set_xticklabels(geneslabels)
		#     plt.ylabel("MSE-"+yname)
		#     plt.xlabel('Genes')
		#     if ntimepoints==0 and ntimeseries==0:
		#         namefig = outputpath+"/Genes-MSE-"+ yname + "_" + str(ngenes)+"genes"
		#     else:
		#         namefig = outputpath+"/Genes-MSE-"+ yname + "_" +str(ngenes)+"genes_"+str(ntimepoints)+"timepoints_"+str(ntimeseries)+"timeseries"
		#     plt.savefig(namefig)
		#     #plt.show()
		return mse, std_mse

	def get_corr(self, yname, ngenes, y, y_pred, outputpath=""):
		# y_act_pred_corr_ss = np.zeros((self.numgenes, 2))
		# y_act_pred_corr_ts = np.zeros((self.numgenes, 2))
		# y_act_pred_corr_future_time_points = np.zeros((self.numgenes, 2))

		corr_arr = np.zeros((ngenes, 2))

		spear_corr_arr = np.zeros((ngenes, 2))

		n_actual_genes = ngenes
		if self.flag_print:
			print("\n")
		#Debug
		#print "yname", yname
		if ngenes>1:
			cum_corr = 0
			spear_cum_corr = 0
			for i in range(0,ngenes):
				#Debug
				#print "Gene: ", i
				actual_y = np.copy(np.asarray(y[i,:]))
				pred_y = np.copy(np.asarray(y_pred[i,:]))

				#Don't correlate conditions where a given gene has value 0s
				where_zeros = np.where(np.asarray(actual_y)==0)[0]
				if where_zeros.size > 0:
					actual_y = np.asarray(np.delete(np.asarray(actual_y), where_zeros))
					pred_y = np.asarray(np.delete(np.asarray(pred_y), where_zeros))

				if len(actual_y) > 1 and len(pred_y) > 1:				
					tmp_corr, tmp_pvalue = scipy.stats.pearsonr(actual_y, pred_y)

					spear_corr, spear_pvalue = scipy.stats.spearmanr(actual_y, pred_y)
				else:
					tmp_corr = 0
					tmp_pvalue = 0
					spear_corr = 0
					spear_pvalue = 0
				#Debug
				# print "Corr values: ", tmp_corr, tmp_pvalue
				# print "Spearman Corr values: ", spear_corr, spear_pvalue
				# print "Vect y and y_pred", y[i,:], y_pred[i,:]

				cum_corr += tmp_corr
				spear_cum_corr += spear_corr
				#correlation outputs nan if vector doesn't vary
				# if np.isnan(tmp_corr):
				# 	print y[i,:], y_pred[i,:]
				corr_arr[i] = [tmp_corr, tmp_pvalue]
				spear_corr_arr[i] = [spear_corr, spear_pvalue]
				if tmp_corr == 0:
					n_actual_genes -= 1
		else:
			corr_avg, tmp_pvalue = scipy.stats.pearsonr(y, y_pred)
			corr_arr = [corr_avg, tmp_pvalue]

			spear_corr_avg, spear_pvalue = scipy.stats.pearsonr(y, y_pred)
			spear_corr_arr = [spear_corr_avg, spear_pvalue]

		if ngenes>1 and n_actual_genes>1:
			corr_avg = float(cum_corr) / float(n_actual_genes)
			spear_corr_avg = float(spear_cum_corr) / float(n_actual_genes)
		else:
			corr_avg = 0
			spear_corr_avg = 0


		#Debug
		if self.flag_print:
			print("Inside the get_corr function Correlation AVG for var name "+yname+" across genes= " + str(corr_avg)+"\n")
		# print "corr_arr False", np.any(np.isnan(corr_arr[:,0]))
		# print "corr_arr True", np.all(np.isfinite(corr_arr[:,0]))

		std_corr = np.std(corr_arr[corr_arr[:,0]>0,0])

		if self.flag_print:
			print("Inside the get_corr function Spearman Correlation AVG for var name"+yname+" across genes= " + str(spear_corr_avg)+"\n")

		spear_std_corr = np.std(spear_corr_arr[spear_corr_arr[:,0]>0,0])


		#Plot MSE for each gene
		plt.clf()
		plt.close()
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax1.plot(y, y_pred, 'bo')#, marker='o')
		ax1.set_ylim(ymin=0)
		ax1.set_xlim(xmin=0)
		ax1.set_title(yname)
		#ax1.set_xticklabels(geneslabels)
		plt.ylabel("Prediction")
		plt.xlabel('Actual')
		namefig = outputpath+"/Genes-Corr-"+ yname + "_" +str(ngenes)+"genes"
		if self.flag_print:
			plt.savefig(namefig)
		#plt.show()

		return corr_avg, std_corr, spear_corr_avg, spear_std_corr

	#@profile
	def build_RF_model(self, nthreads, X, y, tree_meth, max_feat, num_trees, time_, datasetname, output_path, name_run, num_ets_lo, priors_data, tuning_genebygene_randomized, time_step, n_iter_search_par, save_models, index_steady_state_new, index_time_points_new, design_mat, delta_vect, res_mat2, auto_meth, gs=None):

		rfmodel = None
		gc.collect()

		rf_start = time.time()

		# self.datatype = datatype

		print_out_string_to_ret = ""

		#Debug
		if self.flag_print:
			print("Write output to file ", output_path+"/_summary_results.txt \n")

		if self.poot:
			print("Write output to file _summary_results.txt inside the output folder \n")
			print("Learning the OutPredict Model ...\n")

		outfile = open(output_path+"/_summary_results.txt",'w')

		if self.flag_print:
			outfile.write("Write Output to file "+output_path+"/_summary_results.txt \n")

		outfile.write("Write output to file _summary_results.txt inside the output folder \n")

		outfile.write(name_run+"\n")


		X_train = X #self.fromdict_toarr(self.dict_X_train)

		# X_test = self.fromdict_toarr(self.dict_X_test)


		y_train = y #self.fromdict_toarr(self.dict_y_train)

		# y_test = self.fromdict_toarr(self.dict_y_test)

		# num_lo_points = y_test.shape[1]

		num_features = X_train.shape[1]#X_test.shape[2]

		if self.flag_print:
			print("The number of features are: ", num_features)

			outfile.write("The number of features are: "+str(num_features)+"\n")

			print("X_train dim: ", X_train.shape)

			outfile.write("X_train dim: "+str(X_train.shape)+"\n")

			# print "X_test dim: ", X_test.shape

			# outfile.write("X_test dim: "+str(X_test.shape)+"\n")

			print("y_train dim: ", y_train.shape)

			outfile.write("y_train dim: "+str(y_train.shape)+"\n")

			# print "y_test dim: ", y_test.shape

			# outfile.write("y_test dim: "+str(y_test.shape)+"\n")

		#h = hpy()
		#print "Model: "+str(h.heap())+"\n"
		#outfile.write("Model: "+str(h.heap())+"\n")
		#Specific Object
		#print "Model object: "+str(h.iso(self).domisize)+"\n"
		#outfile.write("Model object: "+str(h.iso(self).domisize)+"\n")

		#h.iso(object).domisize

		rfmodel = RF_model(datasetname, self.data_type, self.data_type_lo, self.path_dataset, self.genelist, self.tflist, X_train, self.prior_file, self.prior_type, y_train, num_features, gs, tuning_genebygene_randomized, self.X_test_ss, self.X_test_ts, self.x_test_ts_current_timepoint, self.deltas, self.tau, time_step, save_models, self.flag_print, self.rnd_seed)
		#rfmodel = RF_model(datasetname, self.path_dataset, self.datatype, self.genelist, self.tflist, X_train, X_test, self.prior_file, self.prior_type, y_train, y_test, num_lo_points, num_features, gs)


		#If Timeseries compute MSE for stay the same
		# if self.datatype == "TS":
		# 	y_test_stsNaive = self.get_stsXtestvalues_4timeseries(self.dict_X_test)


		# 	mse_test_stsNaive = self.get_mse("Naive", self.numgenes, y_test, y_test_stsNaive, outputpath=output_path)
		# 	outfile.write("Naive - stay the same time series prediction - MSE test: "+str(mse_test_stsNaive)+"\n")



		VIM, y_train_pred, y_test_pred_ss, y_test_pred_ts, y_test_pred_future_time_points, oob_score_avg, oob_mse_avg = rfmodel.rf_comb(nthreads, output_path, index_steady_state_new, index_time_points_new, design_mat, delta_vect, res_mat2, regulators='all', tree_method=tree_meth, num_of_feats=max_feat, ntrees=num_trees, n_iter_search=n_iter_search_par)
		# VIM, y_test_pred, y_train_pred, y_hat_stdDev_trees_arr, y_act_pred_corr, y_act_pred_corr2 = rfmodel.rf_comb(output_path, num_ets_lo, regulators='all',
		#                                                       tree_method=tree_meth, num_of_feats=max_feat, ntrees=num_trees, h=time_)


		#h = hpy()
		#print "Model: "+str(h.heap())+"\n"
		#outfile.write("Model: "+str(h.heap())+"\n")
		#Specific Object
		#print "RF Model object: "+str(h.iso(rfmodel).domisize)+"\n"
		#outfile.write("RF Model object: "+str(h.iso(rfmodel).domisize)+"\n")

		#Compute MSE for train, test. Based on the user input, MSE might be related to TimeSeries, SteadyState or TS & SS combined.

		mse_train, std_mse_train = self.get_mse("Train", self.numgenes, y_train.transpose(), y_train_pred, outputpath=output_path)

		if self.flag_print:
			outfile.write("MSE train: "+str(mse_train)+" "+str(std_mse_train)+"\n")
			print("MSE train: "+str(mse_train)+" "+str(std_mse_train)+"\n")

		if self.data_type == "SS" or (self.data_type == "TS-SS" and (self.data_type_lo=="TS-SS" or self.data_type_lo=="SS")):
			mse_test_ss, std_mse_ss = self.get_mse("Test_SS", self.numgenes, self.y_test_ss.transpose(), y_test_pred_ss, outputpath=output_path)

			if auto_meth:
				print_out_string_to_ret += "MSE Steady State Test Set: "+str(mse_test_ss)+"\n"
			else:
				print("MSE Steady State Test Set: "+str(mse_test_ss)+"\n")

				outfile.write("MSE Steady State Test Set: "+str(mse_test_ss)+"\n")

			if self.flag_print:
				print("MSE StdDev of avg across genes - Steady State Test Set: "+str(std_mse_ss)+"\n")

				outfile.write("MSE StdDev of avg across genes - Steady State Test Set: "+str(std_mse_ss)+"\n")


			mean_corr_test_ss, std_corr_test_ss, spear_mean_corr_test_ss, spear_std_corr_test_ss = self.get_corr("Corr_Test_SS", self.numgenes, self.y_test_ss.transpose(), y_test_pred_ss, outputpath=output_path)

			if self.flag_print:
				outfile.write("SS: Mean-Std Correlation Coefficient between Actual and Predicted Steady State values: "+str(mean_corr_test_ss)+", "+str(std_corr_test_ss)+" Spearman Corr: "+str(spear_mean_corr_test_ss)+", "+str(spear_std_corr_test_ss)+"\n")

		if self.data_type == "TS" or (self.data_type == "TS-SS" and (self.data_type_lo=="TS-SS" or self.data_type_lo=="TS")):
			mse_test_ts, std_mse_ts = self.get_mse("Test_TS", self.numgenes, self.y_test_ts.transpose(), y_test_pred_ts, outputpath=output_path)
			mse_test_ts_naive, std_mse_ts_naive = self.get_mse("Naive_Test_TS_future_timepoint", self.numgenes, self.y_test_ts_future_timepoint.transpose(), self.x_test_ts_current_timepoint.transpose(), outputpath=output_path)
			mse_test_timepoint0, std_mse_test_timepoint0 = self.get_mse("Timepoint0_Test_TS_future_timepoint", self.numgenes, self.y_test_ts_future_timepoint.transpose(), self.x_test_ts_timepoint0.transpose(), outputpath=output_path)
			mse_test_future_timepoint, std_mse_future_timepoint = self.get_mse("Test_TS_future_timepoint", self.numgenes, self.y_test_ts_future_timepoint.transpose(), y_test_pred_future_time_points, outputpath=output_path)

			#Debug
			diff_test_ts = np.mean((self.y_test_ts.transpose() - y_test_pred_ts))
			diff_test_future_timepoint = np.mean((self.y_test_ts_future_timepoint.transpose() - y_test_pred_future_time_points))
			if self.flag_print:
				print("diff_test_ts", diff_test_ts)
				print("diff_test_future_timepoint", diff_test_future_timepoint)

			if self.flag_print:
				np.savetxt(output_path+'/y_test_ts_future_timepoint_'+str(self.tau)+'.txt', np.asarray(self.y_test_ts_future_timepoint), delimiter=' ')
				np.savetxt(output_path+'/y_test_pred_future_time_points_'+str(self.tau)+'.txt', np.asarray(y_test_pred_future_time_points), delimiter=' ')

			mse_naive_condswise = (np.square(self.y_test_ts_future_timepoint - self.x_test_ts_current_timepoint)).mean(axis=1)
			if self.flag_print:
				print("\n"+"Len conds"+str(len(mse_naive_condswise)))
				print("\n"+"Naive mean standard deviation: "+str(np.mean(mse_naive_condswise)), str(np.std(mse_naive_condswise)))


			N = self.y_test_ts_future_timepoint.transpose().shape[1]  # number of points
			A = self.y_test_ts_future_timepoint.transpose()
			B = self.x_test_ts_current_timepoint.transpose()

			out_corr = np.ones(N)
			out_corr_pval = np.ones(N)

			for i in range(N):
				arr_corr_tmp = scipy.stats.pearsonr(A[:, i], B[:, i])
				out_corr[i] = arr_corr_tmp[0]
				out_corr_pval[i] = arr_corr_tmp[1]

			if self.flag_print:
				print("\n" + "Len conds" + str(N))
				print("\n"+"CORR Naive mean standard deviation: "+str(np.mean(out_corr)), str(np.std(out_corr)))
				print("\n"+"CORR PVALUE Naive mean standard deviation: "+str(np.mean(out_corr_pval)), str(np.std(out_corr_pval)))

			#Look at MSE of each time-series
			# print self.y_test_ts_future_timepoint.transpose().shape
			# print y_test_pred_future_time_points.shape
			# print "MSE of each time-series"
			# for r in range(0,y_test_pred_future_time_points.shape[1]):
			# 	a_tmp = self.y_test_ts_future_timepoint.transpose()[:,r]
			# 	b_tmp = y_test_pred_future_time_points[:,r]
			# 	a_tmp = a_tmp.reshape(self.numgenes,1)
			# 	b_tmp = b_tmp.reshape(self.numgenes,1)
			# 	try:
			# 		mse_test_future_timepoint_singleTimeSeries = self.get_mse("Test_TS_future_timepoint", self.numgenes, a_tmp, b_tmp, outputpath=output_path)
			# 		b_tmp = self.y_test_pred_future_time_pointsx_test_ts_current_timepoint.transpose()[:,r]
			# 		b_tmp = b_tmp.reshape(self.numgenes,1)
			# 		mse_test_ts_naive_singleTimeSeries = self.get_mse("Naive_Test_TS_future_timepoint", self.numgenes, a_tmp, b_tmp, outputpath=output_path)
			# 		outfile.write("Naive "+str(mse_test_ts_naive_singleTimeSeries)+" "+str(r)+" "+"Single Time-series MSE: "+str(mse_test_future_timepoint_singleTimeSeries)+"\n")
			# 		improv_tmp = 100*((float(mse_test_ts_naive_singleTimeSeries) - float(mse_test_future_timepoint_singleTimeSeries)) / float(mse_test_ts_naive_singleTimeSeries))
			# 		print "Improvement", r, improv_tmp
			# 		outfile.write("Improvement "+str(r)+" "+str(improv_tmp)+"\n")
			# 	except:
			# 		pass
			#End Look at MSE of each time-series


			if self.flag_print:
				outfile.write("MSE test Time Series: " + str(mse_test_ts) + " " + str(std_mse_ts) + "\n")
				outfile.write("MSE Timepoint0 test Time Series: " + str(mse_test_timepoint0) + " " + str(
					std_mse_test_timepoint0) + "\n")
				print("MSE test Time-Series Change Prediction for ODE-log model: " + str(mse_test_ts) + " " + str(std_mse_ts) + "\n")
				print("MSE Timepoint0 test Time Series: " + str(mse_test_timepoint0) + " " + str(
					std_mse_test_timepoint0) + "\n")
				print("MSE for Time-Series is Prediction of Last Time-Points which is also called Future Timepoints")
				if auto_meth:
					print("MSE Pen. Value Naive Time-Series Test Set: " + str(mse_test_ts_naive) + "\n")
					outfile.write("MSE Pen. Value Naive  Time-Series Test set: " + str(mse_test_ts_naive) + "\n")

					print("MSE Time-Series Test Set: " + str(mse_test_future_timepoint) + "\n")
					outfile.write("MSE Time-Series Test Set: " + str(mse_test_future_timepoint) + "\n")


			if auto_meth:
				print_out_string_to_ret += "MSE Pen. Value Naive Time-Series Test set: " + str(mse_test_ts_naive) + "\n"
				print_out_string_to_ret += "MSE Time-Series Test Set of the best model is: " + str(mse_test_future_timepoint) + "\n"

			else:
				print("MSE Pen. Value Naive Time-Series Test Set: " + str(mse_test_ts_naive) + "\n")
				outfile.write("MSE Pen. Value Naive  Time-Series Test set: " + str(mse_test_ts_naive) + "\n")


				print("MSE Time-Series Test Set: " + str(mse_test_future_timepoint) + "\n")
				outfile.write("MSE Time-Series Test Set: " + str(mse_test_future_timepoint) + "\n")



			if self.flag_print:
				print("MSE StdDev of avg across genes - Time-Series Test Set: " + str(std_mse_future_timepoint) + "\n")
				outfile.write("MSE StdDev of avg across genes - Time-Series Test Set: " + str(std_mse_future_timepoint) + "\n")

				print("MSE NAIVE StdDev of avg across genes -  Time-Series Test Set: " + str(std_mse_ts_naive) + "\n")
				outfile.write(
					"MSE NAIVE StdDev of avg across genes -  Time-Series Test Set: " + str(std_mse_ts_naive) + "\n")

			mean_corr_test_ts, std_corr_test_ts, spear_mean_corr_test_ts, spear_std_corr_test_ts = None, None, None, None
			mean_corr_naive_test_ts, std_corr_naive_test_ts, spear_mean_corr_naive_test_ts, spear_std_corr_naive_test_ts = None, None, None, None
			mean_corr_t0_test_ts, std_corr_t0_test_ts, spear_mean_corr_t0_test_ts, spear_std_corr_t0_test_ts = None, None, None, None
			mean_corr_test_future_timepoint, std_corr_test_future_timepoint, spear_mean_corr_test_future_timepoint, spear_std_corr_test_future_timepoint = None, None, None, None

			if self.y_test_ts.shape[0]>1:
				mean_corr_test_ts, std_corr_test_ts, spear_mean_corr_test_ts, spear_std_corr_test_ts = self.get_corr("Corr_Test_TS", self.numgenes, self.y_test_ts.transpose(), y_test_pred_ts, outputpath=output_path)
				mean_corr_naive_test_ts, std_corr_naive_test_ts, spear_mean_corr_naive_test_ts, spear_std_corr_naive_test_ts  = self.get_corr("Corr_NAIVE_Test_TS_future_timepoint", self.numgenes, self.y_test_ts_future_timepoint.transpose(), self.x_test_ts_current_timepoint.transpose(), outputpath=output_path)
				mean_corr_t0_test_ts, std_corr_t0_test_ts, spear_mean_corr_t0_test_ts, spear_std_corr_t0_test_ts  = self.get_corr("Corr_Timepoint0_Test_TS_future_timepoint", self.numgenes, self.y_test_ts_future_timepoint.transpose(), self.x_test_ts_timepoint0.transpose(), outputpath=output_path)
				mean_corr_test_future_timepoint, std_corr_test_future_timepoint, spear_mean_corr_test_future_timepoint, spear_std_corr_test_future_timepoint = self.get_corr("Corr_Test_TS_future_timepoint", self.numgenes, self.y_test_ts_future_timepoint.transpose(), y_test_pred_future_time_points, outputpath=output_path)

			if self.flag_print:
				outfile.write("TS: Mean-Std Correlation Coefficient between Actual and Predicted Time series values: "+str(mean_corr_test_ts)+", "+str(std_corr_test_ts)+" Spearman Corr: "+str(spear_mean_corr_test_ts)+", "+str(spear_std_corr_test_ts)+"\n")
				print("TS: Mean-Std Correlation Coefficient between Actual and Predicted Time series values: "+str(mean_corr_test_ts)+", "+str(std_corr_test_ts)+" Spearman Corr: "+str(spear_mean_corr_test_ts)+", "+str(spear_std_corr_test_ts)+"\n")
				if time_step:
					outfile.write(":::::::::::::::::::Since this model uses the Time-Step Approach, so TS Mean-Std equal to MSE Time-Series Last Time-Points Test Set/TS Future Timepoints Mean-Std"+"\n")

				print("TS Naive: Mean-Std Correlation Coefficient between Actual and NAIVE Predicted Time series values: "+str(mean_corr_naive_test_ts)+", "+str(std_corr_naive_test_ts)+" Spearman Corr: "+str(spear_mean_corr_naive_test_ts)+", "+str(spear_std_corr_naive_test_ts)+"\n")

				print("TS Timepoint0 (initial expression levels at t0 as predictions): Mean-Std Correlation Coefficient between Actual and t0 Predicted Time series values: "+str(mean_corr_t0_test_ts)+", "+str(std_corr_t0_test_ts)+" Spearman Corr: "+str(spear_mean_corr_t0_test_ts)+", "+str(spear_std_corr_t0_test_ts)+"\n")

				print("TS Future Timepoints: Mean-Std Correlation Coefficient between Actual and Predicted Future Timepoints values: "+str(mean_corr_test_future_timepoint)+", "+str(std_corr_test_future_timepoint)+" Spearman Corr: "+str(spear_mean_corr_test_future_timepoint)+", "+str(spear_std_corr_test_future_timepoint)+"\n")

				print("TS Naive: Mean-Std Correlation Coefficient between Actual and NAIVE Predicted Time series values: "+str(mean_corr_naive_test_ts)+", "+str(std_corr_naive_test_ts)+" Spearman Corr: "+str(spear_mean_corr_naive_test_ts)+", "+str(spear_std_corr_naive_test_ts)+"\n")

				print("TS Timepoint0 (initial expression levels at t0 as predictions): Mean-Std Correlation Coefficient between Actual and t0 Predicted Time series values: "+str(mean_corr_t0_test_ts)+", "+str(std_corr_t0_test_ts)+" Spearman Corr: "+str(spear_mean_corr_t0_test_ts)+", "+str(spear_std_corr_t0_test_ts)+"\n")

				print("TS Future Timepoints: Mean-Std Correlation Coefficient between Actual and Predicted Future Timepoints values: "+str(mean_corr_test_future_timepoint)+", "+str(std_corr_test_future_timepoint)+" Spearman Corr: "+str(spear_mean_corr_test_future_timepoint)+", "+str(spear_std_corr_test_future_timepoint)+"\n")


		if self.data_type == "TS-SS" and self.data_type_lo=="TS-SS":
			y_test_ss_and_ts = np.concatenate((self.y_test_ss.transpose(), self.y_test_ts_future_timepoint.transpose()), axis=1)

			y_test_pred_ss_and_ts = np.concatenate((y_test_pred_ss, y_test_pred_future_time_points), axis=1)

			mse_test_ss_and_ts, std_mse_test_ss_and_ts = self.get_mse("Test_SS_and_TS_future_timepoint", self.numgenes, y_test_ss_and_ts, y_test_pred_ss_and_ts, outputpath=output_path)
			# mse_test = self.get_mse("Test_TS", self.numgenes, y_test, y_test_pred, outputpath=output_path)


			if self.flag_print:
				print("MSE for Time-Series is Prediction of Last Time-Points which is also called Future Timepoints")


			if auto_meth:
				print_out_string_to_ret += "MSE avg between Steady State and Time-Series Test Set: "+str(mse_test_ss_and_ts)+"\n"
			else:
				outfile.write("MSE avg between Steady State and Time-Series Test Set: "+str(mse_test_ss_and_ts)+"\n")

				print("MSE avg between Steady State and Time-Series Test Set: "+str(mse_test_ss_and_ts)+"\n")

			if self.flag_print:
				outfile.write(
					"MSE StdDev of avg across genes - Avg between Steady State and Time-Series Test Set: " + str(
						std_mse_test_ss_and_ts) + "\n")

				print("MSE StdDev of avg across genes - Avg between Steady State and Time-Series Test Set: " + str(
					std_mse_test_ss_and_ts) + "\n")

			mean_corr_test_ss_and_ts, std_corr_test_ss_and_ts, spear_mean_corr_test_ss_and_ts, spear_std_corr_test_ss_and_ts = self.get_corr("Corr_Test_SS_and_TS_future_timepoint", self.numgenes, y_test_ss_and_ts, y_test_pred_ss_and_ts, outputpath=output_path)

			if self.flag_print:
				outfile.write("SS and TS Future Timepoints: Mean-Std Correlation Coefficient between Actual and Predicted Steady State and future timepoints values together: "+str(mean_corr_test_ss_and_ts)+", "+str(std_corr_test_ss_and_ts)+" Spearman Corr: "+str(spear_mean_corr_test_ss_and_ts)+", "+str(spear_std_corr_test_ss_and_ts)+"\n")

				print("SS and TS Future Timepoints: Mean-Std Correlation Coefficient between Actual and Predicted Steady State and future timepoints values together: "+str(mean_corr_test_ss_and_ts)+", "+str(std_corr_test_ss_and_ts)+" Spearman Corr: "+str(spear_mean_corr_test_ss_and_ts)+", "+str(spear_std_corr_test_ss_and_ts)+"\n")

		# outfile.write("MSE test: "+str(mse_test)+"\n")

		#Debug
		# print "y_test_ss False", np.any(np.isnan(self.y_test_ss))

		# print "y_test_ss True", np.all(np.isfinite(self.y_test_ss))

		# print "y_test_pred_ss False", np.any(np.isnan(y_test_pred_ss))

		# print "y_test_pred_ss True", np.all(np.isfinite(y_test_pred_ss))


		# print "y_test_ts False", np.any(np.isnan(self.y_test_ts))

		# print "y_test_ts True", np.all(np.isfinite(self.y_test_ts))

		# print "y_test_pred_ts False", np.any(np.isnan(y_test_pred_ts))

		# print "y_test_pred_ts True", np.all(np.isfinite(y_test_pred_ts))




		#Debug
		#print "y_test_ss_and_ts", y_test_ss_and_ts.shape #number of genes x number of points
		#print "y_test_pred_ss_and_ts", y_test_pred_ss_and_ts.shape #number of genes x number of points

		# mean_corr = np.mean(y_act_pred_corr[:,0])

		# mean_pvalue = np.mean(y_act_pred_corr[:,1])

		# std_corr = np.std(y_act_pred_corr[:,0])

		# std_pvalue = np.std(y_act_pred_corr[:,1])

		# outfile.write("Mean-Std Correlation Coefficient between Actual and Predicted values: "+str(mean_corr)+", "+str(std_corr)+"\n")

		# outfile.write("Mean-Std Pvalue Correlation Coefficient between Actual and Predicted values: "+str(mean_pvalue)+", "+str(std_pvalue)+"\n")


		# if num_ets_lo>1:

		# 	mean_corr2 = np.mean(y_act_pred_corr2[:,0])

		# 	mean_pvalue2 = np.mean(y_act_pred_corr2[:,1])

		# 	std_corr2 = np.std(y_act_pred_corr2[:,0])

		# 	std_pvalue2 = np.std(y_act_pred_corr2[:,1])

		# 	outfile.write("Mean-Std Correlation Coefficient between Actual and Predicted values(correct): "+str(mean_corr2)+", "+str(std_corr2)+"\n")

		# 	outfile.write("Mean-Std Pvalue Correlation Coefficient between Actual and Predicted values(correct): "+str(mean_pvalue2)+", "+str(std_pvalue2)+"\n")



		# outfile.write("MSE test: "+str(mse_test)+"\n")

		#rank_filename = output_path+"/_ranking_"+str(self.numgenes)+"_genes.txt"
		#aupr = rfmodel.get_link_list(VIM, rank_filename, output_path)


		confidences = pd.DataFrame(VIM, index = self.genelist, columns = self.tflist )

		#The following line prints the matrix of edges score.
		if not(auto_meth):
			confidences.to_csv(output_path+"/Matrix_TF_gene"+str(self.tau)+".tsv", sep="\t")
			#Print ranked list of edges
			import csv
			List = [('TF', 'Target', 'Importance')]
			for source in confidences.columns.values:
				for target in confidences.index.values:
					List.append((source, target, confidences[source][target]))
			with open(output_path+"/Ranked_list_TF_gene"+str(self.tau)+".csv", "w") as f:
				writer = csv.writer(f)
				writer.writerows(List)


		filename_prcurve = "_PRcurve_"+str(datasetname)+str(self.numgenes)+"genes"

		aupr = 0
		if not(auto_meth):
			net_inf = NetworkInference(self.flag_print, self.rnd_seed)
			aupr, random_aupr = net_inf.summarize_results(output_path, filename_prcurve, confidences, gs, True)
		else:
			net_inf = NetworkInference(self.flag_print, self.rnd_seed)
			aupr, random_aupr = net_inf.summarize_results(output_path, filename_prcurve, confidences, gs, False)

		if auto_meth:
			print_out_string_to_ret += "\n Influences Inference: Causal connections from transcription factors to genes are printed to file as a ranked list of interactions (Ranked_list_TF_gene) and as a matrix (Matrix_TF_gene) inside the output folder \n"
		else:
			print("\n Influences Inference: Causal connections from transcription factors to genes are printed to file as a ranked list of interactions (Ranked_list_TF_gene.csv) and as a matrix (Matrix_TF_gene.tsv) inside the output folder \n")
			outfile.write("\n Influences Inference: Causal connections from transcription factors to genes are printed to file as a ranked list of interactions (Ranked_list_TF_gene.csv) and as a matrix (Matrix_TF_gene.tsv) inside the output folder \n")


		if self.flag_print and not(auto_meth):

			print("Area under Precision-Recall based on goldstandard: ", aupr)

			outfile.write("Area under Precision-Recall based on goldstandard: "+str(aupr)+"\n")

			outfile.write("Random AUPR: "+str(random_aupr)+"\n")


		gc.collect()
		rf_end = time.time()
		total_time = rf_end - rf_start
		if self.flag_print:
			print("OutPredict over: %.2f seconds" %(total_time))

			outfile.write("OutPredict over: "+str(total_time)+" seconds:"+"\n")

		# if not(auto_meth):
		# 	outfile.close()

		mse_to_ret = 0
		mse_corr_to_ret = 0
		if self.data_type == "TS-SS" and self.data_type_lo == "TS-SS":
			mse_to_ret = mse_test_ss_and_ts
			mse_corr_to_ret = mean_corr_test_ss_and_ts
		elif self.data_type == "SS" or (self.data_type == "TS-SS" and self.data_type_lo == "SS"):
			mse_to_ret = mse_test_ss
			mse_corr_to_ret = mean_corr_test_ss
		else:
			mse_to_ret = mse_test_future_timepoint
			mse_corr_to_ret = mean_corr_test_future_timepoint

		return oob_mse_avg, oob_score_avg, aupr, mse_to_ret, mse_corr_to_ret, print_out_string_to_ret, confidences, gs, outfile
