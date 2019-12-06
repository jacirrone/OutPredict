"""
Dataset Class Implementation for preprocessing the dataset given as input

"""



import pandas as pd
import re
import os
import numpy as np
from collections import OrderedDict
from Preprocess import Preprocess
import matplotlib.pyplot as plt
import pickle



class Dataset:

	def __init__(self, datasetname, rnd_seed):
		#print("Constructor Dataset Class")
		#print("Preprocessing the dataset "+datasetname+" ...")
		self.datasetname = datasetname

		self.flag_print = None
		self.parse_4dyng3 = None
		self.auto_meth = None
		self.poot = None


		np.random.seed(rnd_seed)
		self.rnd_seed = rnd_seed


	def str_to_bool(self, s):
		if s == "True":
			return True
		elif s == "False":
			return False
		else:
			raise ValueError



	def readDatasetFromMetaDataFile(self, meta_data):
		start_index_values_ts = meta_data[meta_data['is1stLast'] == "f"].index.values
		TS_vectors = []

		num_total_timeseries_points = 0

		for first_ind in start_index_values_ts:
			is1stLast = ''
			ind = first_ind
			timepoint = 0
			ts = OrderedDict()
			while is1stLast!='l':
				is1stLast = meta_data.loc[ind, 'is1stLast']

				condName = meta_data.loc[ind, 'condName']
				delt = float(meta_data.loc[ind, 'del.t'])

				if np.isnan(delt):
					delt = 0

				timepoint = timepoint + delt

				ts[condName] = timepoint
				ind += 1

				num_total_timeseries_points += 1

			TS_vectors.append(ts)

		index_steady_state = meta_data[meta_data['isTs'] == False].index.values
		steady_state_cond = meta_data.loc[index_steady_state, 'condName'].values


		return TS_vectors, steady_state_cond, index_steady_state, num_total_timeseries_points



	def choose_timeseries_LO_lastPoints_random_withTimehorizon(self, percent, num_total_timeseries_points, TS_vectors, timehorizon):
		num_of_timeseries = len(TS_vectors)
		lopoints_y = OrderedDict()
		lopoints_x = OrderedDict()
		t0_lopoints = OrderedDict()
		np.random.seed(self.rnd_seed)
		number_of_lo_points = int(np.floor(percent * num_total_timeseries_points))#num_of_timeseries))
		
		if number_of_lo_points<=num_of_timeseries:
			timeseries_indices = np.random.choice(num_of_timeseries, number_of_lo_points, replace = False)
			#Debug
			#print "number_of_lo_points<=num_of_timeseries"
		else:
			timeseries_indices = list(range(0,num_of_timeseries))
			#Debug
			#print "num_of_timeseries", num_of_timeseries
		
		for ind in timeseries_indices:
			ts_tmp = TS_vectors[ind]

			lo_y_key = list(ts_tmp.keys())[-1]
			tp_y = ts_tmp[lo_y_key]
			try:
				if self.flag_print:
					print("Third to last time point: ", list(ts_tmp.keys())[-4])
				lo_x_key = list(ts_tmp.keys())[-1-timehorizon]
				tp_x = ts_tmp[lo_x_key]
				lopoints_x[lo_x_key] = tp_x
				lopoints_y[lo_y_key] = tp_y
				#Get t = 0 for each leave out time point
				t0_key = list(ts_tmp.keys())[0]
				t0 = ts_tmp[t0_key]
				t0_lopoints[t0_key] = t0
			except:
				if self.flag_print:
					print("The timeseries with index"+str(ind)+"is too short for leaving out the last time point")
					print("Also, OutPredict does NOT allow to leave out last time points of time series with less than four time points")
					print(ts_tmp)
		return lopoints_x, lopoints_y, t0_lopoints, timeseries_indices



	def choose_LO_timeseries_random_withTimehorizon(self, number_of_ts_to_lo, TS_vectors, timehorizon):
		num_of_timeseries = len(TS_vectors)

		ts_lo_y = OrderedDict()
		ts_lo_x = OrderedDict()
		np.random.seed(self.rnd_seed)

		timeseries_indices = np.random.choice(num_of_timeseries, number_of_ts_to_lo, replace = False)


		for ind in timeseries_indices:
			ts_tmp = TS_vectors[ind]
			lo_y_key = list(ts_tmp.keys())[timehorizon:]
			try:
				lo_x_key = list(ts_tmp.keys())[:-timehorizon]
				num_of_points = len(lo_y_key)

				for j in range(0, num_of_points):
					single_lo_x_key = lo_x_key[j]
					single_lo_y_key = lo_y_key[j]

					tp_y = ts_tmp[single_lo_y_key]
					tp_x = ts_tmp[single_lo_x_key]
					ts_lo_x[single_lo_x_key] = tp_x
					ts_lo_y[single_lo_y_key] = tp_y
			except:
				if self.flag_print:
					print("The timeseries at index"+str(ind)+"is too short for leaving out the last time point")
					print(ts_tmp)

		return ts_lo_x, ts_lo_y, timeseries_indices



	def choose_steadystate_LO_points_random(self, percent, steady_state_cond):
		num_of_ss = len(steady_state_cond)
		number_of_lo_points = int(np.floor(percent * num_of_ss))
			
		np.random.seed(self.rnd_seed)

		ss_lo_indices = (np.random.choice(num_of_ss, number_of_lo_points, replace = False))

		ss_lo_cond_names = steady_state_cond[ss_lo_indices]

		return ss_lo_cond_names, ss_lo_indices



	def get_X_y_for_timeseries_withTimehorizon(self, TS_vectors, timehorizon, expression, tf_names, ts_lopoints_y, lopoints_bool):

		#Time horizon; if time horizon is equal 1 the total number of points are: total_num_time_points - num_timeseries

		num_tfs = len(tf_names)

		dict_X_ts = OrderedDict() #key is gene name, value is the corresponding X_ts
		dict_y_ts = OrderedDict() #key is gene name, value is the corresponding y_ts

		gene_names = expression.index.values

		for genename in gene_names:
			X_ts = []
			y_ts = []

			#for loop over the timeseries
			for ts_tmp in TS_vectors:
				#for loop over a single timeseries

				#DEBUG
				#print ts_tmp

				ind = 0
				for key_x in list(ts_tmp.keys()):
					try:
						key_y =  list(ts_tmp.keys())[ind + timehorizon]
						#print "keyx", key_x, "key_y", key_y

						if (key_y in list(ts_lopoints_y.keys())) == lopoints_bool:

							expression_tfs = np.copy(expression.loc[:num_tfs, key_x].values)
							expression_x_gene = np.copy(expression.loc[genename, key_x])

							X_ts_tmp = np.append(expression_tfs, expression_x_gene)
							X_ts.append(X_ts_tmp)


							expression_y_gene = np.copy(expression.loc[genename, key_y])

							y_ts.append(expression_y_gene)
					except:
						continue
					ind += 1

			#Matrix ready to be used by the random forest model
			X_ts = np.asarray(X_ts, dtype=float)
			y_ts = np.asarray(y_ts, dtype=float)

			dict_X_ts[genename] = X_ts
			dict_y_ts[genename] = y_ts

		#DEBUG
		# print dict_X_ts
		# print dict_y_ts

		return dict_X_ts, dict_y_ts



	def get_X_y_for_steadystate(self, steady_state_cond, expression, tf_names, ss_lo_cond_names):

		num_tfs = len(tf_names)

		dict_X_ss = OrderedDict() #key is gene name, value is the corresponding X_ts
		dict_y_ss = OrderedDict() #key is gene name, value is the corresponding y_ts

		gene_names = expression.index.values

		for genename in gene_names:
			X_ss = []
			y_ss = []

			#for loop over the timeseries
			for ss_tmp in steady_state_cond:

				if (ss_tmp in ss_lo_cond_names) == False:

					expression_tfs = np.copy(expression.loc[:num_tfs, ss_tmp].values)

					if genename in tf_names:
						indx_tf = list(tf_names).index(genename)
						expression_tfs[indx_tf] = 0

					X_ss.append(expression_tfs)

					expression_y_gene = np.copy(expression.loc[genename, ss_tmp])

					y_ss.append(expression_y_gene)


			#Matrix ready to be used by random forest
			X_ss = np.asarray(X_ss, dtype=float)
			y_ss = np.asarray(y_ss, dtype=float)

			dict_X_ss[genename] = X_ss
			dict_y_ss[genename] = y_ss

		#DEBUG
		#print dict_X_ts
		#print dict_y_ts

		return dict_X_ss, dict_y_ss



	def proper_concat_twolists(self, first_list, total_list_df):
		#Debug
		# print "first_list", len(first_list)
		# print "total_list_df", total_list_df.shape
		# print "intersection ", len(np.intersect1d(first_list, total_list_df[0]))

		other_genes = total_list_df[~total_list_df.isin(first_list)]
		other_genes.dropna(inplace=True)
		other_genes.reset_index(inplace=True)
		other_genes.drop('index', axis=1, inplace=True)
		# print "first_list", len(first_list)

		# print "other_genes", len(other_genes)
		ret_list = np.concatenate((first_list, other_genes[0]))

		#return list: first_list + (total_list not in first_list)
		return ret_list



	def intersect_twolists(self, list1, list2_):

		other_genes = total_list_df[~total_list_df.isin(first_list)]
		other_genes.dropna(inplace=True)
		other_genes.reset_index(inplace=True)
		other_genes.drop('index', axis=1, inplace=True)

		ret_list = np.concatenate((first_list, other_genes[0]))

		return ret_list



	def loadData(self, input_dir, name_run, script_dir, data_type, data_type_lo, delTmax, delTmin, tau, tfa_bool, timehorizon, percent_LO_points, num_ets_lo, time_step, thres_coeff_var, prior_type, prior_file):

		str_output = ""
		uniq_dups = []

		np.random.seed(self.rnd_seed)
		pps = Preprocess(self.rnd_seed)

		pps.delTmax = delTmax
		pps.delTmin = delTmin
		pps.tau = tau
		pps.input_dir = input_dir
		pps.str_output = str_output
		pps.flag_print = self.flag_print
		pps.priors_file = prior_file

		#IF CONDITIONS HAVE DUPLICATED NAMES, PRINT A META DATA FILE CALLED "meta_data_uniq.tsv" with only unique conds
		metadata_1 = pps.input_dataframe(pps.meta_data_file, has_index=False, strict=False)
		num_dups_conds = len(metadata_1.condName[metadata_1.condName.duplicated(keep=False)])

		if num_dups_conds>0:
			uniq_dups = (metadata_1.condName[metadata_1.condName.duplicated(keep=False)]).unique()
			num_uniq_dups = len(uniq_dups)
			if self.flag_print:
				print("name of duplicated conds in meta data: ", num_dups_conds)
				print("number of unique in dups conds", num_uniq_dups)
			metadata_1.set_index(['condName'], inplace=True)

			metadata_1_series = metadata_1.groupby(level=0).cumcount()
			metadata_1_series = "repet" + metadata_1_series.astype(str)
			metadata_1.index =  metadata_1.index + metadata_1_series.replace('repet0','')
			#metadata_1.index = metadata_1.index + "_dup_"+ metadata_1.groupby(level=0).cumcount().astype(str).replace('0','')

			#The following code is to fix names of prevCol for duplicated conditions
			metadata_copy = metadata_1.copy()
			name_prev_cond = np.nan
			count = 0
			for index, row in (metadata_1[metadata_1.isTs==True]).iterrows():
				if (row['is1stLast']=='m') or (row['is1stLast']=='l'):
					if row['prevCol'] != name_prev_cond:
						if self.flag_print:
							print(index, row)
						metadata_copy.at[index, 'prevCol'] = name_prev_cond
						count = count + 1
				name_prev_cond = index

			if self.flag_print:
				print(count)
			if count != num_dups_conds-num_uniq_dups:
				raise ValueError('Wrong meta data format')

			#metadata_copy.drop(['Unnamed: 0'], axis=1, inplace=True)
			metadata_copy.reset_index(inplace=True)
			metadata_copy.columns = ['condName', 'isTs', 'is1stLast', 'prevCol', 'del.t']
			cols = ['isTs', 'is1stLast', 'prevCol', 'del.t', 'condName']
			metadata_copy = metadata_copy[cols]

			pps.meta_data_file = "meta_data_uniq.tsv"
			path_file = pps.input_path(pps.meta_data_file)
			# metadata_copy.is1stLast = '"' + metadata_copy.is1stLast + '"'
			# metadata_copy.prevCol = '"' + metadata_copy.prevCol + '"'
			# metadata_copy.condName = '"' + metadata_copy.condName + '"'
			# metadata_copy.columns = ['"isTs"', '"is1stLast"', '"prevCol"', '"del.t"', '"condName"']
			metadata_copy.to_csv(path_file, sep="\t", index=False, na_rep='NA')#, quoting=csv.QUOTE_NONE)

			#Add to expression file duplicated conds, this is important for how the leave-out section is implemented
			expression_1 = pps.input_dataframe(pps.expression_matrix_file, has_index=False, strict=False)
			count = 0
			for ud in uniq_dups:
				pattern = re.compile(ud+"repet"+"\d")
				for cond_tmp in metadata_copy.condName:
					if pattern.match(cond_tmp):
						expression_1[cond_tmp] = expression_1[ud]
						count = count + 1

			if count != num_dups_conds-num_uniq_dups:
				raise ValueError('Wrong expression/meta_data format')

			col_arr = (np.asarray(expression_1.columns[1:]))
			expression_1.columns = np.insert(col_arr, 0, "")
			pps.expression_matrix_file = "expression_new.tsv"
			path_file = pps.input_path(pps.expression_matrix_file)
			expression_1.to_csv(path_file, sep="\t", index=False, na_rep='NA')#, quoting=csv.QUOTE_NONE)

		#END CODE FOR PRINTING NEW UNIQUE META DATA FILE AND NEW EXPRESSION FILE


		str_output = pps.get_data(thres_coeff_var, str_output, prior_type)

		pps.compute_common_data(uniq_dups, time_step)


		#CODE FOR LEAVE OUT DATA
		TS_vectors, steady_state_cond, index_steady_state, num_total_timeseries_points = self.readDatasetFromMetaDataFile(pps.meta_data)


		#Parse data to dynGenie3 format in case parse_4dyng3 is set to "True"

		# print pps.expression_matrix.head()
		# print pps.expression_matrix.index.tolist()
		# print pps.expression_matrix.loc["G1", :]

		if self.parse_4dyng3:
			#(TS_data,time_points,genes,TFs,alphas)

			# import sys
			# reload(sys)
			# sys.setdefaultencoding('utf8')
			print("Start parsing data to dynGenie3 format")
			TS_data = list()
			time_points = list()
			genes = pps.expression_matrix.index.tolist()
			genes = np.asarray(genes).astype(str)
			genes = genes.tolist()
			num_gene_names = len(genes)
			alphas = [0.02]*num_gene_names
			alphas = np.asarray(alphas).astype(float)
			alphas = alphas.tolist()

			for ts_tmp in TS_vectors:
				#for loop over a single timeseries

				ts_tmp_vect = list(ts_tmp.keys())

				num_time_points_intstmp = len(ts_tmp_vect)

				ts_dynGenie3 = np.zeros((num_time_points_intstmp, num_gene_names))
				ts_dynGenie3 = np.transpose(pps.expression_matrix.loc[:, ts_tmp_vect])
				TS_data.append(np.asarray(ts_dynGenie3))

				time_points_i = np.zeros(num_time_points_intstmp)

				for j, key in enumerate(ts_tmp_vect):
					time_points_i[j] = np.float(ts_tmp[key])

				time_points.append(time_points_i)

			# print TS_data
			# print type(TS_data[1])

			SS_data = np.transpose(pps.expression_matrix[steady_state_cond])

			#(TS_data,time_points,genes,TFs,alphas)
			TFs = np.asarray(pps.tf_names).astype(str)
			TFs = TFs.tolist()

			TS_data_file = "TS_data.pkl"
			path_file = pps.input_path(TS_data_file)
			with open(path_file, 'wb') as f:
				pickle.dump([TS_data, time_points, genes, TFs, alphas], f)
			# cPickle.dump(TS_data, f)
			# print type(TS_data)
			# cPickle.dump(time_points, f)
			# print type(time_points)
			# cPickle.dump(alphas, f)
			# print type(alphas)
			# cPickle.dump(genes, f)
			# print type(genes)
			f.close()
			# with open(output_path_estimators+'/Gene'+str(output_idx), 'rb') as f:
			#     treeEstimator = cPickle.load(f)
			SS_data_file = "SS_data.txt"
			path_file = pps.input_path(SS_data_file)
			SS_data.to_csv(path_file, sep="\t", index=False, na_rep='NA')
			print("End parsing data to dynGenie3 format")
			# # #END parse data to dynGenie3 format

		#Debug
		# pps.design.to_csv(os.path.abspath(os.path.join(pps.input_dir))+"/_design.txt", sep="\t")
		# pps.response.to_csv(os.path.abspath(os.path.join(pps.input_dir))+"/_response.txt", sep="\t")
		# pps.meta_data.to_csv(os.path.abspath(os.path.join(pps.input_dir))+"/_meta_data.txt", sep="\t")

		if data_type == "TS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="TS")):
			if num_ets_lo > 0:
				ts_lopoints_x, ts_lopoints_y, timeseries_indices_lo = self.choose_LO_timeseries_random_withTimehorizon(num_ets_lo, TS_vectors, timehorizon)
			else:
				ts_lopoints_x, ts_lopoints_y, t0_lopoints, timeseries_indices_lo = self.choose_timeseries_LO_lastPoints_random_withTimehorizon(percent_LO_points, num_total_timeseries_points, TS_vectors, timehorizon)


		if data_type == "SS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="SS")):
			ss_lo_cond_names = list()
			ss_lo_cond_names = np.asarray(ss_lo_cond_names)
			ss_lo_indices = list()
			ss_lo_indices = np.asarray(ss_lo_indices)

			if len(steady_state_cond) > 0:
				ss_lo_cond_names, ss_lo_indices = self.choose_steadystate_LO_points_random(percent_LO_points, steady_state_cond)

		#Debug
		# print "num_total_timeseries_points", num_total_timeseries_points
		# print "len(ss_lo_cond_names)", len(steady_state_cond)
		# print "len(pps.meta_data)", len(pps.meta_data)


		#TS_vectors, steady_state_cond, index_steady_state, num_total_timeseries_points
		# TS_vectors [OrderedDict([('S0_1', 0),
		#               ('S1_1', 60.0),
		#               ('S2_1', 120.0),
		#               ('S3_1', 180.0),
		#               ('S4_1', 240.0),
		#               ('S5_1', 300.0),
		#               ('S6_1', 360.0)]),
		#  OrderedDict([('S0_2', 0),
		#               ('S1_2', 60.0),
		#               ('S2_2', 120.0),
		#               ('S3_2', 180.0),
		#               ('S4_2', 240.0),
		#               ('S5_2', 300.0),
		#               ('S6_2', 360.0)]),......]
		# steady_state_cond
		# array(['LBexp_1', 'LBexp_2', 'LBexp_3',....]

		# index_steady_state
		# array([163, 164, 165, 166, 167,....]

		# num_total_timeseries_points
		# 163


		#Leave-out Time-series points
		#ts_lopoints_x, ts_lopoints_y, timeseries_indices_lo
		# timeseries_indices_lo left out
		# array([31, 15, 26, 17])
		# ts_lopoints_x, ts_lopoints_y
		# OrderedDict([('MG+90_2', 95.0), ('SMM_1', 0), ('dia5_3', 5.0), ('SMM_3', 0)])
		# OrderedDict([('MG+120_2', 125.0), ('Salt_1', 10.0), ('dia15_3', 15.0), ('Salt_3', 10.0)])

		#Leave-out Steady state points
		#ss_lo_cond_names, ss_lo_indices
		# array(['H2O2_1', 'LBGexp_2', 'LBtran_2', ....]
		# array([100,  10,   4,  81,  97,  65, ... ]


		if self.flag_print:
			print("Shape of design var before leaving-out data: ", str(pps.design.shape))
			print("Shape of response var before leaving-out data: ", str(pps.response.shape))

		str_output = str_output+"Shape of design var before leaving-out data: "+str(pps.design.shape)+"\n"
		str_output = str_output+"Shape of response var before leaving-out data: "+str(pps.response.shape)+"\n"

		#Debug
		# w = csv.writer(open("ts_lopoints_x.csv", "w"))
		# for key, val in ts_lopoints_x.items():
		#     w.writerow([key, val])

		# pps.design.to_csv(os.path.abspath(os.path.join(pps.input_dir))+"/_design.txt", sep="\t")
		# pps.response.to_csv(os.path.abspath(os.path.join(pps.input_dir))+"/_response.txt", sep="\t")


		#Before splitting the dataset in training and test, check if want to learn on SS only or TS only
		if data_type == "SS":
			str_output = str_output+"::::::::STEADY-STATE ONLY - LOOK AT JUST THE SHAPES OF DESIGN AND RESPONSE VARIABLES"+"\n"
			only_steady_state_indxes = (pps.design.columns.isin(steady_state_cond))
			pps.design = pps.design.loc[:, only_steady_state_indxes]#, axis=1, inplace=True)
			pps.response = pps.response.loc[:, only_steady_state_indxes]#, axis=1, inplace=True)
			pps.half_tau_response = pps.half_tau_response.loc[:, only_steady_state_indxes]

			pps.delta_vect = pps.delta_vect.loc[:, (pps.delta_vect.columns.isin(steady_state_cond))]#, axis=1, inplace=True)

		if data_type == "TS":
			str_output = str_output+"::::::::TIME-SERIES ONLY - LOOK AT JUST THE SHAPES OF DESIGN AND RESPONSE VARIABLES"+"\n"
			pps.design.drop(steady_state_cond, axis=1, inplace=True)
			pps.response.drop(steady_state_cond, axis=1, inplace=True)
			pps.half_tau_response.drop(steady_state_cond, axis=1, inplace=True)

			pps.delta_vect.drop(steady_state_cond, axis=1, inplace=True)

		# print "Shape of design design before splitting: "+str(pps.design.shape)
		# print "Shape of response response before splitting: "+str(pps.response.shape)
		#
		# design_tmp = pps.design
		# tfs_tmp = list(set(pps.tf_names).intersection(pps.expression_matrix.index))
		# X_tmp = np.asarray(design_tmp.loc[tfs_tmp,:].values)
		# X_tmp = (X_tmp - (X_tmp.mean(axis=1)).reshape(-1,1)) / (X_tmp.std(axis=1)).reshape(-1,1)
		# design_tmp_2 = pd.DataFrame(X_tmp ,index = tfs_tmp, columns = design_tmp.columns)
		# pps.design = design_tmp_2
		#
		# print "Shape of design after normalization/standardization: ", pps.design.shape
		#
		# response_tmp = pps.response
		# Y_tmp = np.asarray(response_tmp.values)
		# Y_tmp = (Y_tmp - (Y_tmp.mean(axis=1)).reshape(-1,1)) / (Y_tmp.std(axis=1)).reshape(-1,1)
		# response_tmp_2 = pd.DataFrame(Y_tmp ,index = response_tmp.index, columns = response_tmp.columns)
		# pps.response = response_tmp_2
		#
		# print "Shape of response after normalization/standardization: ", pps.response.shape

		if data_type == "SS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="SS")):
			#Leaving out Steady state points
			pps.leave_out_ss_design = pps.design[ss_lo_cond_names]
			pps.design.drop(ss_lo_cond_names, axis=1, inplace=True)
			pps.leave_out_ss_response = pps.response[ss_lo_cond_names]
			pps.response.drop(ss_lo_cond_names, axis=1, inplace=True)
			pps.half_tau_response.drop(ss_lo_cond_names, axis=1, inplace=True)
			if self.flag_print:
				print("Shape of leave out SS design var: ", pps.leave_out_ss_design.shape)
				print("Shape of leave out SS response var: ", pps.leave_out_ss_response.shape)

			pps.delta_vect.drop(ss_lo_cond_names, axis=1, inplace=True)


		if data_type == "TS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="TS")):
			#Leaving out Time series points
			pps.leave_out_ts_design = pps.design[list(ts_lopoints_x.keys())]
			pps.design.drop(list(ts_lopoints_x.keys()), axis=1, inplace=True)
			pps.leave_out_ts_response = pps.response[list(ts_lopoints_x.keys())]
			pps.response.drop(list(ts_lopoints_x.keys()), axis=1, inplace=True)
			pps.half_tau_response.drop(list(ts_lopoints_x.keys()), axis=1, inplace=True)
			if self.flag_print:
				print("Shape of leave out TS design var: ", pps.leave_out_ts_design.shape)
				print("Shape of leave out TS response var: ", pps.leave_out_ts_response.shape)

			pps.delta_vect.drop(list(ts_lopoints_x.keys()), axis=1, inplace=True)

		if self.flag_print:
			print("Shape of design var after leaving-out data: ", pps.design.shape)
			print("Shape of response var after leaving-out data: ", pps.response.shape)

		str_output = str_output+ "Shape of design var after leaving-out data: "+str(pps.design.shape)+"\n"
		str_output = str_output+ "Shape of response var after leaving-out data: "+str(pps.response.shape)+"\n"

		if data_type == "SS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="SS")):
			str_output = str_output+ "Shape of leave out SS design var: "+str(pps.leave_out_ss_design.shape)+"\n"
			str_output = str_output+ "Shape of leave out SS response var: "+str(pps.leave_out_ss_response.shape)+"\n"

		if data_type == "TS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="TS")):
			str_output = str_output+ "Shape of leave out TS design var: "+str(pps.leave_out_ts_design.shape)+"\n"
			str_output = str_output+ "Shape of leave out TS response var: "+str(pps.leave_out_ts_response.shape)+"\n"

		#END CODE FOR LEAVE OUT DATA

		if data_type == "SS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="SS")):
			steady_state_cond_new = list(steady_state_cond.copy())
			for element in ss_lo_cond_names:
				steady_state_cond_new.remove(element)
		else:
			steady_state_cond_new = steady_state_cond

		index_steady_state_new = []
		indexes_all = list(range(0,len(pps.design.columns)))
		delta_vect = list()
		#Debug
		#print len(indexes_all)
		if data_type == "SS" or data_type == "TS-SS":
			for element in steady_state_cond_new:
				index_steady_state_new.append(pps.design.columns.get_loc(element))
			index_steady_state_new = np.asarray(index_steady_state_new)

		index_time_points_new = []
		if data_type == "TS" or data_type == "TS-SS":
			index_time_points_new = set(indexes_all) - set(index_steady_state_new)
			index_time_points_new = np.asarray(list(index_time_points_new))

		#Debug
		#print len(index_time_points_new)
		#print len(index_steady_state_new)

		#Debug
		# print "pps.priors_data.shape", pps.priors_data.shape
		# print "len(pps.priors_data.abs().sum(axis=0))", len(pps.priors_data.abs().sum(axis=0))
		# print "len(pps.priors_data.abs().sum(axis=0))", len(pps.priors_data.abs().sum(axis=1))
		# print "len(pps.priors_data.sum(axis=0))", len(pps.priors_data.sum(axis=0))
		# print "type(np.abs(pps.priors_data))", type(np.abs(pps.priors_data))
		# pps.priors_data.to_csv(os.path.abspath(os.path.join(pps.input_dir))+"/_ppspriors_data.txt", sep="\t")
		# pps.gold_standard.to_csv(os.path.abspath(os.path.join(pps.input_dir))+"/_ppsgold_standard.txt", sep="\t")
		# print type(pps.gold_standard)
		# pps.design.to_csv(os.path.abspath(os.path.join(pps.input_dir))+"/_design.txt", sep="\t")
		# pps.response.to_csv(os.path.abspath(os.path.join(pps.input_dir))+"/_response.txt", sep="\t")

		if prior_type == "binary_all":
			num_edges_prior = np.sum(pps.priors_data.values != 0)
		num_edges_gs = np.sum(pps.gold_standard.values != 0)
		if self.flag_print:
			if prior_type == "binary_all":
				print("Number of edges in the prior: ", num_edges_prior, pps.priors_data.shape)
			print("Number of edges in the evaluation part of the gold standard: ", num_edges_gs, pps.gold_standard.shape)
		if prior_type == "binary_all":
			str_output = str_output+ "Number of edges in the prior: "+str(num_edges_prior)+str(pps.priors_data.shape)+"\n"
		str_output = str_output+ "Number of edges in the evaluation part of the gold standard: "+str(num_edges_gs)+str(pps.gold_standard.shape)+"\n"

		# print "pps.activity.shape", pps.activity.shape
		# print pps.expression_matrix.shape
		# print len(pps.tf_names)
		# print pps.gold_standard.shape
		# print pps.response.shape

		if tfa_bool:
			#compute_activity()
			# """
			# Compute Transcription Factor Activity
			# """
			if self.flag_print:
				print('Computing Transcription Factor Activity ... ')
			tfs = list(set(pps.tf_names).intersection(pps.expression_matrix.index))
			#TFA_calculator = TFA(pps.priors_data, pps.design, pps.half_tau_response, tfs)
			pps.activity = pps.compute_transcription_factor_activity(tfs)
			#pps.activity, pps.priors_data= TFA_calculator.compute_transcription_factor_activity()

		else:
			if self.flag_print:
				print('Using just expression, NO Transcription Factor Activity')
			expression_matrix = pps.design
			tfs = list(set(pps.tf_names).intersection(pps.expression_matrix.index))
			activity = pd.DataFrame(expression_matrix.loc[tfs,:].values,
									index = tfs,
									columns = expression_matrix.columns)
			if self.flag_print:
				print(('Design matrix of shape: {}'.format(activity.shape)))
			pps.activity = activity

		tf_names = pps.activity.index.tolist()#pps.priors_data.columns #pps.tf_names

		#Leave-out SS
		if data_type == "SS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="SS")):
			expression_matrix_lo_ss = pps.leave_out_ss_design
			leave_out_ss_design = pd.DataFrame(expression_matrix_lo_ss.loc[tf_names,:].values,
											   index = tf_names,
											   columns = expression_matrix_lo_ss.columns)
			pps.leave_out_ss_design = leave_out_ss_design

		#Leave-out TS
		if data_type == "TS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="TS")):
			expression_matrix_lo_ts = pps.leave_out_ts_design
			leave_out_ts_design = pd.DataFrame(expression_matrix_lo_ts.loc[tf_names,:].values,
											   index = tf_names,
											   columns = expression_matrix_lo_ts.columns)
			pps.leave_out_ts_design = leave_out_ts_design


		expression = pps.expression_matrix#this is the initial one but then there is filtering and stuff


		goldstandard = pps.gold_standard
		genelist = pps.response.index.tolist()#pps.expression_matrix.index.tolist()
		numtfs = len(tf_names)


		X = pps.activity.transpose().values#X [n_samples, n_features]
		y = pps.response.transpose().values#y [n_samples, num_genes]

		if self.flag_print:
			print("Shape of design var X: "+str(X.shape))
			print("Shape of response var Y: "+str(y.shape))
		str_output = str_output+ "Shape of design var X: "+str(X.shape)+"\n"
		str_output = str_output+ "Shape of response var Y: "+str(y.shape)+"\n"

		if self.flag_print:
			print("X False", np.any(np.isnan(X)))

			print("X True", np.all(np.isfinite(X)))

			print("y False", np.any(np.isnan(y)))

			print("y True", np.all(np.isfinite(y)))

		X = np.float64(X)

		y = np.float64(y)

		output_path = script_dir+"/output/"+name_run+"_numgenes"+str(len(genelist))+"_numtfs"+str(numtfs)

		if not os.path.exists(output_path):
			os.makedirs(output_path)
		# else:
		# 	if self.poot or not(self.auto_meth):
		# 		num_folders = len([name for name in os.listdir(script_dir+"/output/") if
		# 							   os.path.isdir(os.path.join(script_dir+"/output/",name)) and (name_run+"_numgenes"+str(len(genelist))+"_numtfs"+str(numtfs)) in name])
		# 		os.makedirs(output_path + "_" + str(num_folders))
		# 		output_path = output_path + "_" + str(num_folders)

		if prior_type == "binary_all":
			if not os.path.exists(input_dir+"/priors"):
				os.makedirs(input_dir+"/priors")

		if prior_type == "binary_all":
			#Save plot of prior number of targets for each TF distribution
			priors_data_tmp = np.abs(pps.priors_data)
			index_tmp = priors_data_tmp.sum(axis=0)!=0
			prior_num_tfs = np.sum(index_tmp)
			#Debug print TFs 
			#print priors_data_tmp.columns[index_tmp]
			#Debug #print priors_data_tmp.sum(axis=0)[index_tmp]
			max_outdegree = np.max(priors_data_tmp.sum(axis=0)[index_tmp])
			#Debug #print "max_outdegree", max_outdegree
			max_outdegree = np.int(max_outdegree)
			out_prior_tfs_outdegrees = "Num of TFs in prior: "+str(prior_num_tfs)+" Mean and var of targets for TFs in prior: "+str(np.mean(priors_data_tmp.sum(axis=0)[index_tmp]))+" , "+str(np.std(priors_data_tmp.sum(axis=0)[index_tmp]))
			str_output = str_output + out_prior_tfs_outdegrees + "\n"
			ax = priors_data_tmp.sum(axis=0)[index_tmp].plot(kind="hist", bins=list(range(0,max_outdegree+1)))
			ax.set_title("Prior outdegrees distribution")
			ax.set_xlabel("outdegree of TFs ( i.e. TFs num of targets)")
			if self.flag_print:
				plt.savefig(output_path+"/Prior outdegrees distribution_numTFs"+str(prior_num_tfs)+"_numEdges"+str(num_edges_prior))
			plt.close()

		#Save plot of Eval GS number of targets for each TF distribution
		gold_standard_tmp = np.abs(pps.gold_standard)
		index_tmp2 = gold_standard_tmp.sum(axis=0)!=0
		gs_num_tfs = np.sum(index_tmp2)
		max_outdegree2 = np.max(gold_standard_tmp.sum(axis=0)[index_tmp2])
		max_outdegree2 = np.int(max_outdegree2)
		#Debug #print gold_standard_tmp.sum(axis=0)[index_tmp2]
		#Debug #print max_outdegree2
		out_gs_tfs_outdegrees = "Num of TFs in eval gold standard: "+str(gs_num_tfs)+" Mean and var of targets for TFs in eval GS: "+str(np.mean(gold_standard_tmp.sum(axis=0)[index_tmp2]))+" , "+str(np.std(gold_standard_tmp.sum(axis=0)[index_tmp2]))
		str_output = str_output + out_gs_tfs_outdegrees + "\n"
		#Debug print TFs 
		#print gold_standard_tmp.columns[index_tmp2]
		ax1 = gold_standard_tmp.sum(axis=0)[index_tmp2].plot(kind="hist", bins=list(range(0,max_outdegree2+1)))
		ax1.set_title("Eval Gold standard outdegrees distribution")
		ax1.set_xlabel("outdegree of TFs ( i.e. TFs num of targets)")
		if self.flag_print:
			plt.savefig(output_path+"/Eval Gold standard outdegrees distribution_numTFs"+str(gs_num_tfs)+"_numEdges"+str(num_edges_gs))
		plt.close()

		if prior_type == "binary_all":
			#Write gold standard priors to file
			pps.priors_data.to_csv(input_dir+"/priors/"+prior_file, sep="\t")


		if self.flag_print:
			outfile = open(output_path+"/_preprocessing.txt",'w')
			outfile.write("Run name: "+str(name_run)+"\n")
			outfile.write(str_output)

		if data_type == "SS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="SS")):
			if len(steady_state_cond) > 0:
				#Debug
				if self.flag_print:
					print("Leave-out points for steady state: ", ss_lo_cond_names, ss_lo_indices)
					outfile.write("Leave-out points for steady state: "+str(ss_lo_cond_names)+str(ss_lo_indices)+"\n")

		if data_type == "TS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="TS")):
			if self.flag_print:
				print("Leave-out points for timeseries: ", ts_lopoints_x, ts_lopoints_y, timeseries_indices_lo)
				outfile.write("Leave-out points for timeseries: "+str(ts_lopoints_x)+str(ts_lopoints_y)+str(timeseries_indices_lo)+"\n")


		# print "New dimensions after coeff of var filter..."
		# outfile.write("New dimensions after coeff of var filter... \n")
		if self.flag_print:
			print("Expression dim: ", expression.shape)
			outfile.write("Expression dim: "+str(expression.shape)+"\n")
		if self.flag_print:
			print("Num of tfs: ", len(tf_names))
			outfile.write("Num of tfs: "+str(len(tf_names))+"\n")
		if self.flag_print:
			print("Num of genes: ", len(genelist))
			outfile.write("Num of genes: "+str(len(genelist))+"\n")
		if self.flag_print:
			if prior_type == "binary_all":
				print("Priors dim: ", pps.priors_data.shape)
				outfile.write("Priors dim: "+str(pps.priors_data.shape)+"\n")
		if self.flag_print:
			print("Goldstandard dim: ", goldstandard.shape)
			outfile.write("Goldstandard dim: "+str(goldstandard.shape)+"\n")


		#Print INFO to log file
		if self.flag_print:
			print("The number of genes is: ", len(genelist))
			outfile.write("The number of genes is: "+str(len(genelist))+"\n")
		if self.flag_print:
			print("The number of TFs is: ", len(tf_names))
			outfile.write("The number of TFs is: "+str(len(tf_names))+"\n")
		if self.flag_print:
			print("The total Number of data points in the dataset is: ", len(pps.meta_data))
			outfile.write("The total Number of data points in the dataset is: "+str(len(pps.meta_data))+"\n")
		if self.flag_print:
			print("The total number of time series is: ", len(TS_vectors))
			outfile.write("The total number of time series is: "+str(len(TS_vectors))+"\n")
		if self.flag_print:
			print("The number of total time points is: ", num_total_timeseries_points)
			outfile.write("The number of total time points is: "+str(num_total_timeseries_points)+"\n")
		if self.flag_print:
			print("The number of total steady state points is: ", len(steady_state_cond))
			outfile.write("The number of total steady state points is: "+str(len(steady_state_cond))+"\n")

		if data_type == "SS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="SS")):
			if self.flag_print:
				print("The percentage of leave-out steady state points is: ", str(100*float(len(ss_lo_indices))/len(steady_state_cond)))
				outfile.write("The percentage of leave-out steady state points is: "+str(100*float(len(ss_lo_indices))/len(steady_state_cond))+"\n")

		if data_type == "TS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="TS")):
			if self.flag_print:
				print("The percentage of leave-out time series points is: ", str(100*float(len(timeseries_indices_lo))/num_total_timeseries_points))
				outfile.write("The percentage of leave-out time series points is: "+str(100*float(len(timeseries_indices_lo))/num_total_timeseries_points)+"\n")
				outfile.close()

		#All variables that can be returned if necessary
		# (All points)
		# TS_vectors, steady_state_cond, num_total_timeseries_points

		# #Training and leave out points
		# index_time_points_new, index_steady_state_new, pps.leave_out_ss_design(X_test_ss), pps.leave_out_ss_response, pps.leave_out_ts_design, pps.leave_out_ts_response

		# #leave out points
		# ss_lo_cond_names, ts_lopoints_x, ts_lopoints_y, timeseries_indices_lo

		if data_type == "SS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="SS")):

			X_test_ss = pps.leave_out_ss_design.transpose().values

			y_test_ss = pps.leave_out_ss_response.transpose().values
		else:
			X_test_ss = ""
			y_test_ss = ""


		deltas = []
		if data_type == "TS" or (data_type == "TS-SS" and (data_type_lo=="TS-SS" or data_type_lo=="TS")):
			X_test_ts = pps.leave_out_ts_design.transpose().values

			y_test_ts = pps.leave_out_ts_response.transpose().values

			ts_lopoints_y_keys = list(ts_lopoints_y.keys())

			for i,k in enumerate(ts_lopoints_x.keys()):
				# #Debug
				# #print "ts_lopoints_x[k]", ts_lopoints_x[k]
				# if float((ts_lopoints_x[k])) == 0:
				# 	log_of_frac = 1
				# else:
				# 	#No log
				# 	#log_of_frac = float(ts_lopoints_y[ts_lopoints_y_keys[i]]) / float((ts_lopoints_x[k]))
				#
				# 	log_of_frac = np.log(float(ts_lopoints_y[ts_lopoints_y_keys[i]]) / float((ts_lopoints_x[k])))
				#deltas.append(log_of_frac)

				#Original
				deltas.append(ts_lopoints_y[ts_lopoints_y_keys[i]] - (ts_lopoints_x[k]))

			y_test_ts_future_timepoint = pps.expression_matrix.loc[genelist, ts_lopoints_y_keys].transpose().values

			x_test_ts_current_timepoint = pps.expression_matrix.loc[genelist, list(ts_lopoints_x.keys())].transpose().values

			x_test_ts_timepoint0 = pps.expression_matrix.loc[genelist, list(t0_lopoints.keys())].transpose().values

		else:
			X_test_ts = ""
			y_test_ts = ""
			y_test_ts_future_timepoint = ""
			x_test_ts_current_timepoint = ""
			x_test_ts_timepoint0 = ""



		#Debug
		#print y_test_ts_future_timepoint
		#print x_test_ts_current_timepoint

		return X, y, genelist, tf_names, goldstandard, output_path, pps.priors_data, X_test_ss, X_test_ts, y_test_ss, y_test_ts, x_test_ts_current_timepoint, y_test_ts_future_timepoint, deltas, x_test_ts_timepoint0, index_steady_state_new, index_time_points_new, pps.design, pps.delta_vect, pps.res_mat2


