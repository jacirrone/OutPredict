"""
Preprocess Class Implementation

"""

import os
import scipy
import numpy as np
import pandas as pd


class Preprocess(object):
    input_dir = None
    expression_matrix_file = "expression.tsv"  # "expression_new.tsv"#"expression.tsv"
    tf_names_file = "tf_names.tsv"
    meta_data_file = "meta_data.tsv"  # "meta_data_uniq.tsv"#"meta_data.tsv"
    leave_out_meta_data_file = "leave_out_meta_data.tsv"
    priors_file = "gold_standard.tsv"
    gold_standard_file = "gold_standard.tsv"
    # random_seed = 42

    # Computed data structures
    expression_matrix = None  # expression_matrix dataframe
    tf_names = None  # tf_names list
    meta_data = None  # meta data dataframe
    leave_out_meta_data = None
    priors_data = None  # priors data dataframe
    gold_standard = None  # gold standard dataframe
    flag_print = None

    def __init__(self, rnd_seed):
        # Do nothing (all configuration is external to init)
        np.random.seed(rnd_seed)
        pass

    def get_data(self, thres_coeff_var, str_output, prior_type):
        """
        Read data files in to data structures.
        """
        if self.flag_print:
            print(self.expression_matrix_file)
        self.expression_matrix = self.input_dataframe(self.expression_matrix_file)

        tf_file = self.input_file(self.tf_names_file)
        self.tf_names = self.read_tf_names(tf_file)

        # Debug
        # print type(self.tf_names)

        if thres_coeff_var >= 0:
            if self.flag_print:
                print("Filter for coefficient of variation")
            # outfile.write("Filter for coefficient of variation \n")
            str_output = str_output + "Filter for coefficient of variation" + "\n"

            genelist = self.expression_matrix.index.tolist()

            if self.flag_print:
                print("num of total genes Before filtering: ", len(genelist))
            # outfile.write("num of total genes Before filtering: "+str(len(genelist))+"\n")
            str_output = str_output + "num of total genes Before filtering: " + str(len(genelist)) + "\n"

            coeffvariation = scipy.stats.variation(self.expression_matrix, axis=1)

            coeffvariation = np.nan_to_num(coeffvariation)

            if self.flag_print:
                print("num of total genes After filtering: ", len(coeffvariation[coeffvariation > thres_coeff_var]))
            # outfile.write("num of total genes After filtering: "+str(len(coeffvariation[coeffvariation>thres_coeff_var]))+"\n")
            str_output = str_output + "num of total genes After filtering: " + str(
                len(coeffvariation[coeffvariation > thres_coeff_var])) + "\n"

            indx_genes_filtered = np.where(coeffvariation <= thres_coeff_var)
            indx_genes_filtered = list(indx_genes_filtered[0])

            if self.flag_print:
                print("num of total genes removed: ", len(indx_genes_filtered))
            # outfile.write("num of total genes removed: "+str(len(indx_genes_filtered))+"\n")
            str_output = str_output + "num of total genes removed: " + str(len(indx_genes_filtered)) + "\n"

            genelist_arr = np.asarray(genelist)

            names_genes_removed = genelist_arr[indx_genes_filtered]

            names_genes_removed_df = pd.DataFrame(names_genes_removed)
            tfs_in_removedgenes = names_genes_removed_df[names_genes_removed_df.isin(self.tf_names)]
            tfs_in_removedgenes.dropna(inplace=True)
            tfs_in_removedgenes.reset_index(inplace=True)
            tfs_in_removedgenes.drop('index', axis=1, inplace=True)

            tfs_in_removedgenes = list(tfs_in_removedgenes[0])

            if self.flag_print:
                print("num of tfs in removed genes: ", len(tfs_in_removedgenes))
            # outfile.write("num of tfs removed in removed genes: "+str(len(tfs_in_removedgenes))+"\n")
            str_output = str_output + "num of tfs in removed genes: " + str(len(tfs_in_removedgenes)) + "\n"

            # Don't remove tfs if thres_coeff_var != 0
            if thres_coeff_var != 0:
                names_genes_removed = np.asarray(list(set(list(names_genes_removed)) - set(tfs_in_removedgenes)))

                # Remove genes in expression, tf_names and goldstandard

            # Remove rows in expression matrix
            self.expression_matrix.drop(names_genes_removed, axis=0, inplace=True)

            # Remove tfs in tf_names if thres_coeff_var != 0 so if it's equal to 0 remove tfs
            tf_names = pd.read_table(self.input_path(self.tf_names_file), header=None)
            tf_names.set_index(0, inplace=True)

            # Don't remove tfs if thres_coeff_var != 0 so if it's equal to 0 remove tfs
            if thres_coeff_var == 0:
                tf_names.drop(tfs_in_removedgenes, axis=0, inplace=True)

            self.tf_names = tf_names.index.tolist()

            if self.flag_print:
                print("num of total tfs after filtering: ", len(self.tf_names))
            # outfile.write("num of total tfs after filtering: "+str(len(tf_names))+"\n")
            str_output = str_output + "num of total tfs after filtering: " + str(len(tf_names)) + "\n"

            # print type(self.tf_names)

        # Read metadata, creating a default non-time series metadata file if none is provided

        if self.flag_print:
            print(self.meta_data_file)

        self.meta_data = self.input_dataframe(self.meta_data_file, has_index=False, strict=False)
        if self.meta_data is None:
            self.meta_data = self.create_default_meta_data(self.expression_matrix)

        # To reach this point, it means that
        # Prior case: prior_type is either gold_standard or steady_state and the file with name prior_file exists
        # No prior case: prior type = "no"

        if prior_type == "binary_all":
            self.priors_data = self.input_dataframe(self.priors_file)
            only_1_and_0 = np.array_equal(self.priors_data.values, self.priors_data.values.astype(bool))
            if not (only_1_and_0):
                print("Error Wrong format prior file for gold standard - please refer to doc")
                exit(1)
            # Filter priors first based on common data with expression, this is esp important when expres and tf_names is filtered with thres_coeff_var
            self.filter_expression_and_priors()

        # If SS or TS weights are priors evaluate on the entire gold standard
        if prior_type == "real_all" or prior_type == "no":
            if not (os.path.exists(self.input_dir + self.gold_standard_file)):
                if self.flag_print:
                    print(
                        ":::::::Gold Standard file (gold_standard.tsv) does NOT exist - Construct Gold Standard Matrix with only 1s - necessary for aupr computation")
                self.gold_standard = pd.DataFrame(1, columns=self.tf_names, index=self.expression_matrix.index.tolist())
            else:
                self.gold_standard = self.input_dataframe(self.gold_standard_file)
            self.filter_expression_and_gs()
        elif prior_type == "binary_all":  # if prior binary then gold standard network equal to prior
            # subdivide 50/50 if same file if gold_standard.tsv is given as priors and you want to do aupr evaluation.
            # str_output = self.set_gold_standard_and_priors(str_output) #gs network 50% of the prior
            self.gold_standard = self.priors_data.copy()
        else:
            print("Error Wrong input data - please refer to doc")
            exit(1)

        return str_output

    # def set_gold_standard_and_priors(self, str_output):
    #     self.priors_data = self.input_dataframe(self.priors_file)
    #     self.gold_standard = self.input_dataframe(self.gold_standard_file)
    #     return str_output

    def set_gold_standard_and_priors(self, str_output):
        """
        Ignore the gold standard file. Instead, create a gold standard
        from a 50/50 split of the prior. Half of original prior becomes the new prior,
        the other half becomes the gold standard
        """

        # Debug
        # print "PriorGoldStandardSplitWorkflowBase class set_gold_standard_and_priors"
        split_ratio = 0.5
        # self.priors_data = self.input_dataframe(self.priors_file)

        num_edges_gs = np.sum(self.priors_data.values != 0)

        if self.flag_print:
            print("Number of edges in the full gold standard: " + str(num_edges_gs))
        str_output = str_output + "Number of edges in the full gold standard: " + str(num_edges_gs) + "\n"
        # Save plot of Full GS number of targets for each TF distribution
        gold_standard_tmp = np.abs(self.priors_data)
        index_tmp2 = self.priors_data.sum(axis=0) != 0
        gs_num_tfs = np.sum(index_tmp2)
        out_gs_tfs_outdegrees = "Num of TFs in Full Gold standard: " + str(
            gs_num_tfs) + " Mean and var of targets for TFs in full GS: " + str(
            np.mean(gold_standard_tmp.sum(axis=0)[index_tmp2])) + " , " + str(
            np.std(gold_standard_tmp.sum(axis=0)[index_tmp2]))
        if self.flag_print:
            print(out_gs_tfs_outdegrees)
        str_output = str_output + out_gs_tfs_outdegrees + "\n"
        # ax1 = gold_standard_tmp.sum(axis=0)[index_tmp2].plot(kind="hist", bins=range(0,len(index_tmp2)))
        # ax1.set_title("Full Gold standard outdegrees distribution")
        # ax1.set_xlabel("outdegree of TFs ( i.e. TFs num of targets)")

        str_output = str_output + "Number of edges in the full gold standard: " + str(
            np.sum(self.priors_data.values != 0)) + "\n"

        prior = pd.melt(self.priors_data.reset_index(), id_vars='index')
        prior_edges = prior.index[prior.value != 0]
        keep = np.random.choice(prior_edges, int(len(prior_edges) * split_ratio), replace=False)
        prior_subsample = prior.copy(deep=True)
        gs_subsample = prior.copy(deep=True)
        prior_subsample.loc[prior_edges[~prior_edges.isin(keep)], 'value'] = 0
        gs_subsample.loc[prior_edges[prior_edges.isin(keep)], 'value'] = 0
        prior_subsample = pd.pivot_table(prior_subsample, index='index', columns='variable', values='value',
                                         fill_value=0)
        gs_subsample = pd.pivot_table(gs_subsample, index='index', columns='variable', values='value', fill_value=0)
        self.priors_data = prior_subsample
        self.gold_standard = gs_subsample
        # Debug
        # print "Number of edges in the prior: ", np.sum(self.priors_data.values != 0)
        # print "Number of edges in the evaluation part of the gold standard: ", np.sum(self.gold_standard.values != 0)
        return str_output

    def input_path(self, filename):
        return os.path.abspath(os.path.join(self.input_dir, filename))

    def create_default_meta_data(self, expression_matrix):
        metadata_rows = expression_matrix.columns.tolist()
        metadata_defaults = {"isTs": "FALSE", "is1stLast": "e", "prevCol": "NA", "del.t": "NA", "condName": None}
        data = {}
        for key in list(metadata_defaults.keys()):
            data[key] = pd.Series(data=[metadata_defaults[key] if metadata_defaults[key] else i for i in metadata_rows])
        return pd.DataFrame(data)

    def input_file(self, filename, strict=True):
        path = self.input_path(filename)
        # Debug
        # print "path", path
        if os.path.exists(path):
            return open(path)
        elif not strict:
            return None
        raise ValueError("no such file " + repr(path))

    def input_dataframe(self, filename, strict=True, has_index=True):
        # Debug
        # print filename, "filename"
        f = self.input_file(filename, strict)
        if f.readline == "" or len(f.readlines()) == 0 or f == None:
            return None
        f = self.input_file(filename, strict)
        if f is not None:
            return self.df_from_tsv(f, has_index)
        else:
            assert not strict
            return None

    def compute_common_data(self, uniq_dups, time_step):
        """
        Compute common data structures like design and response matrices.
        """
        # self.filter_expression_and_priors()
        if self.flag_print:
            print('Creating input and output matrix ... ')
        # self.design_response_driver.delTmin = self.delTmin
        # self.design_response_driver.delTmax = self.delTmax
        # self.design_response_driver.tau = self.tau)
        (self.design, self.response, self.delta_vect, self.res_mat2) = self.input_output_construction(
            self.expression_matrix, self.meta_data, uniq_dups, time_step, self.delTmin, self.delTmax, self.tau)

        # if self.flag_print:
        # compute half_tau_response
        # print('Setting up TFA specific response matrix ... ')
        # self.design_response_driver.tau = self.tau / 2
        (self.design, self.half_tau_response, self.delta_vect, self.res_mat2) = self.input_output_construction(
            self.expression_matrix, self.meta_data, uniq_dups, time_step, self.delTmin, self.delTmax, self.tau / 2)

    def filter_expression_and_priors(self):
        """
        Guarantee that each row of the prior is in the expression and vice versa.
        Also filter the priors to only includes columns, transcription factors, that are in the tf_names list
        """
        exp_genes = self.expression_matrix.index.tolist()
        all_regs_with_data = list(
            set.union(set(self.expression_matrix.index.tolist()), set(self.priors_data.columns.tolist())))
        tf_names = list(set.intersection(set(self.tf_names), set(all_regs_with_data)))
        self.priors_data = self.priors_data.loc[exp_genes, tf_names]
        self.priors_data = pd.DataFrame.fillna(self.priors_data, 0)

    def filter_expression_and_gs(self):
        """
        Guarantee that each row of the gold standard is in the expression and vice versa.
        Also filter the gold standard to only includes columns, transcription factors, that are in the tf_names list
        """
        exp_genes = self.expression_matrix.index.tolist()
        all_regs_with_data = list(
            set.union(set(self.expression_matrix.index.tolist()), set(self.gold_standard.columns.tolist())))
        tf_names = list(set.intersection(set(self.tf_names), set(all_regs_with_data)))
        self.gold_standard = self.gold_standard.loc[exp_genes, tf_names]
        self.gold_standard = pd.DataFrame.fillna(self.gold_standard, 0)

    def read_tf_names(self, file_like):
        "Read transcription factor names from one-column tsv file.  Return list of names."
        exp = pd.read_csv(file_like, sep="\t", header=None)
        assert exp.shape[1] == 1, "transcription factor file should have one column "
        return list(exp[0])

    def df_from_tsv(self, file_like, has_index=True):
        "Read a tsv file or buffer with headers and row ids into a pandas dataframe."
        return pd.read_csv(file_like, sep="\t", header=0, index_col=0 if has_index else False)

    # def metadata_df(self, file_like):
    #     "Read a metadata file as a pandas data frame."
    #     return pd.read_csv(file_like, sep="\t", header=0, index_col="condName")

    def input_output_construction(self, expression_mat, metadata_dataframe, uniq_dups, time_step, delTmin, delTmax,
                                  tau):

        meta_data = metadata_dataframe.copy()
        meta_data = meta_data.replace('NA', np.nan, regex=False)
        exp_mat = expression_mat.copy()

        special_char_dictionary = {'+': 'specialplus', '-': 'specialminus', '.': 'specialperiod', '/': 'specialslash',
                                   '\\': 'special_back_slash', ')': 'special_paren_backward',
                                   '(': 'special_paren_forward', ',': 'special_comma', ':': 'special_colon',
                                   ';': 'special_semicoloon', '@': 'special_at', '=': 'special_equal',
                                   '>': 'special_great', '<': 'special_less', '[': 'special_left_bracket',
                                   ']': 'special_right_bracket', "%": 'special_percent', "*": 'special_star',
                                   '&': 'special_ampersand', '^': 'special_arrow', '?': 'special_question',
                                   '!': 'special_exclamation', '#': 'special_hashtag', "{": 'special_left_curly',
                                   '}': 'special_right_curly', '~': 'special_tilde', '`': 'special_tildesib',
                                   '$': 'special_dollar', '|': 'special_vert_bar'}

        cols = exp_mat.columns.tolist()
        for ch in list(special_char_dictionary.keys()):
            # need this edge case for passing micro test
            if len(meta_data['condName'][~meta_data['condName'].isnull()]) > 0:
                meta_data['condName'] = meta_data['condName'].str.replace(ch, special_char_dictionary[ch])
            if len(meta_data['prevCol'][~meta_data['prevCol'].isnull()]) > 0:
                meta_data['prevCol'] = meta_data['prevCol'].str.replace(ch, special_char_dictionary[ch])
            cols = [item.replace(ch, special_char_dictionary[ch]) for item in cols]
        exp_mat.columns = cols

        cond = meta_data['condName'].copy()
        prev = meta_data['prevCol'].copy()
        delt = meta_data['del.t'].copy()

        ##Debug
        # print "delt", delt
        # print "type", type(delt)

        # Note: when you want to modify delta, e.g.log(delta), change here and in Run.py
        # LOG
        if np.sum(delt <= 1) > 0:
            delt = delt + 1
        delt = np.log(delt)

        # deltas=1
        # delt[~np.isnan(delt)] = 1

        # delTmin = self.delTmin
        # delTmax = self.delTmax
        # tau = self.tau
        prev.loc[delt > delTmax] = np.nan
        delt.loc[delt > delTmax] = np.nan
        not_in_mat = set(cond) - set(exp_mat)
        cond_dup = cond.duplicated()
        if len(not_in_mat) > 0:
            cond = cond.str.replace('[/+-]', '.')
            prev = cond.str.replace('[/+-]', '.')
            if cond_dup != cond.duplicated():
                raise ValueError(
                    'Tried to fix condition names in meta data so that they would match column names in expression matrix, but failed')

        # check if there are condition names missing in expression matrix
        not_in_mat = set(cond) - set(exp_mat)
        if len(not_in_mat) > 0:
            if self.flag_print:
                print(not_in_mat)
            raise ValueError(
                'Error when creating design and response. The conditions printed above are in the meta data, but not in the expression matrix')

        cond_n_na = cond[~cond.isnull()]
        steady = prev.isnull() & ~(cond_n_na.isin(prev.replace(np.nan, "NA")))

        des_mat = pd.DataFrame(exp_mat[cond[steady]])
        res_mat = pd.DataFrame(exp_mat[cond[steady]])
        if not (time_step):
            res_mat2 = pd.DataFrame(exp_mat[cond[steady]])
        else:
            res_mat2 = pd.DataFrame()

        delta_vect = pd.DataFrame(index=[0], columns=exp_mat[cond[steady]].columns)
        curr_time_point_for_time_series = 0

        for i in list(np.where(~steady)[0]):
            following = list(np.where(prev.str.contains(cond[i]) == True)[0])
            following_delt = list(delt[following])

            try:
                off = list(np.where(following_delt[0] < delTmin)[0])
            except:
                off = []

            while len(off) > 0:
                off_fol = list(np.where(prev.str.contains(cond[following[off[0]]]) == True)[0])
                off_fol_delt = list(delt[off_fol])
                # Debug
                # print "list(delt)", list(delt)
                # print "off_fol", len(off_fol)
                # print "delt", len(delt)
                # print "delt[off_fol]", delt[off_fol]
                # print "off_fol_delt", off_fol_delt
                # print "off", len(off)
                # print "following", len(following)
                # print "off[0]", off[0]
                # print "following_delt", len(following_delt)
                # print "off_fol_delt[0]", off_fol_delt[0]
                following = following[:off[0]] + following[off[0] + 1:] + off_fol
                following_delt = following_delt[:off[0]] + following_delt[off[0] + 1:] + [
                    float(off_fol_delt[0]) + float(following_delt[off[0]])]
                off = list(np.where(following_delt < [delTmin])[0])

            n = len(following)
            cntr = 0

            for j in following:
                if n > 1 and cond[i] not in uniq_dups:
                    if self.flag_print:
                        print(cond[i])
                    this_cond = "%s_dupl%02d" % (cond[i], cntr + 1)
                    original_this_cond = this_cond
                    k = 1
                    while this_cond in res_mat.columns:
                        this_cond = original_this_cond + '.{}'.format(int(k))
                        k = k + 1
                else:
                    this_cond = cond[i]

                des_tmp = np.concatenate((des_mat.values, exp_mat[cond[i]].values[:, np.newaxis]), axis=1)
                des_names = list(des_mat.columns) + [this_cond]
                des_mat = pd.DataFrame(des_tmp, index=des_mat.index, columns=des_names)
                # next line defines the response variable, which in the naive way is represented by x(t+delta) only
                if time_step:
                    interp_res = exp_mat[cond[j]].astype(
                        'float64')  # (float(tau)/float(following_delt[cntr])) * (exp_mat[cond[j]].astype('float64') - exp_mat[cond[i]].astype('float64')) + exp_mat[cond[i]].astype('float64')
                else:
                    interp_res = (float(tau) / float(following_delt[cntr])) * (
                            exp_mat[cond[j]].astype('float64') - exp_mat[cond[i]].astype('float64')) + exp_mat[
                                     cond[i]].astype('float64')

                    interp_res2 = exp_mat[cond[j]].astype('float64')
                    res_tmp2 = np.concatenate((res_mat2.values, interp_res2.values[:, np.newaxis]), axis=1)
                    res_names2 = list(res_mat2.columns) + [this_cond]
                    res_mat2 = pd.DataFrame(res_tmp2, index=res_mat2.index, columns=res_names2)

                res_tmp = np.concatenate((res_mat.values, interp_res.values[:, np.newaxis]), axis=1)
                res_names = list(res_mat.columns) + [this_cond]
                res_mat = pd.DataFrame(res_tmp, index=res_mat.index, columns=res_names)

                #
                # if curr_time_point_for_time_series == 0:
                #     log_of_frac = 1
                #     curr_time_point_for_time_series = curr_time_point_for_time_series + float(following_delt[cntr])
                # else:
                #     curr_time_point_for_time_series = curr_time_point_for_time_series + float(following_delt[cntr])
                #
                #     log_of_frac = np.log(float(curr_time_point_for_time_series)/float(following_delt[cntr]))
                #     #No log
                #     #log_of_frac = float(curr_time_point_for_time_series)/float(following_delt[cntr])
                # #print "log_of_frac", log_of_frac
                # delta_vect_tmp = np.concatenate((delta_vect.values,pd.DataFrame([log_of_frac]).values),axis=1)

                # Original
                delta_vect_tmp = np.concatenate((delta_vect.values, pd.DataFrame([float(following_delt[cntr])]).values),
                                                axis=1)

                delta_vect_names = list(delta_vect.columns) + [this_cond]
                delta_vect = pd.DataFrame(delta_vect_tmp, index=delta_vect.index, columns=delta_vect_names)

                cntr = cntr + 1

            # special case: nothing is following this condition within delT.min
            # and it is the first of a time series --- treat as steady state

            if n == 0 and prev.isnull()[i]:
                curr_time_point_for_time_series = 0

                des_mat = pd.concat([des_mat, exp_mat[cond[i]]], axis=1)
                des_mat.rename(columns={des_mat.columns.values[len(des_mat.columns) - 1]: cond[i]}, inplace=True)
                res_mat = pd.concat([res_mat, exp_mat[cond[i]]], axis=1)
                res_mat.rename(columns={res_mat.columns.values[len(res_mat.columns) - 1]: cond[i]}, inplace=True)
                # delta_vect.append("NA")
                delta_vect_tmp = np.concatenate((delta_vect.values, pd.DataFrame([np.nan]).values), axis=1)
                delta_vect_names = list(delta_vect.columns) + [this_cond]
                delta_vect = pd.DataFrame(delta_vect_tmp, index=delta_vect.index, columns=delta_vect_names)

                if not (time_step):
                    res_mat2 = pd.concat([res_mat2, exp_mat[cond[i]]], axis=1)
                    res_mat2.rename(columns={res_mat2.columns.values[len(res_mat2.columns) - 1]: cond[i]}, inplace=True)

        cols_des_mat = des_mat.columns.tolist()
        cols_res_mat = res_mat.columns.tolist()
        cols_delt_mat = delta_vect.columns.tolist()

        special_char_inv_map = {v: k for k, v in list(special_char_dictionary.items())}
        for sch in list(special_char_inv_map.keys()):
            cols_des_mat = [item.replace(sch, special_char_inv_map[sch]) for item in cols_des_mat]
            cols_res_mat = [item.replace(sch, special_char_inv_map[sch]) for item in cols_res_mat]
            cols_delt_mat = [item.replace(sch, special_char_inv_map[sch]) for item in cols_delt_mat]

        des_mat.columns = cols_des_mat
        res_mat.columns = cols_res_mat
        delta_vect.columns = cols_delt_mat

        # Jac - My implementation in Dataset.py takes into account that if there are dups conds...
        # Make the conds unique.
        des_mat = des_mat.loc[:, ~des_mat.columns.duplicated()]
        res_mat = res_mat.loc[:, ~res_mat.columns.duplicated()]

        return (des_mat, res_mat, delta_vect, res_mat2)

    def compute_transcription_factor_activity(self, tfs, allow_self_interactions_for_duplicate_prior_columns=True):
        """
        TFA calculates transcription factor activity using matrix pseudoinverse

            Parameters
        --------
        prior: pd.dataframe
            binary or numeric g by t matrix stating existence of gene-TF interactions.
            g: gene, t: TF.

        expression_matrix: pd.dataframe
            normalized expression g by c matrix. g--gene, c--conditions

        expression_matrix_halftau: pd.dataframe
            normalized expression matrix for time series.

        allow_self_interactions_for_duplicate_prior_columns=True: boolean
            If True, TFs that are identical to other columns in the prior matrix
            do not have their self-interactios removed from the prior
            and therefore will have the same activities as their duplicate tfs.
        """

        # Find TFs that have non-zero columns in the priors matrix
        non_zero_tfs = self.priors_data.columns[(self.priors_data != 0).any(axis=0)].tolist()
        # Delete tfs that have neither prior information nor expression
        delete_tfs = set(self.priors_data.columns).difference(self.priors_data.index).difference(non_zero_tfs)
        # Raise warnings
        if len(delete_tfs) > 0:
            message = " ".join([str(len(delete_tfs)).capitalize(),
                                "transcription factors are removed because no expression or prior information exists."])
            warnings.warn(message)
            self.priors_data = self.priors_data.drop(delete_tfs, axis=1)

        # Create activity dataframe with values set by default to the transcription factor's expression
        # activity = pd.DataFrame(self.design.loc[self.priors_data.columns,:].values,
        #         index = self.priors_data.columns,
        #         columns = self.design.columns)

        # OutPredict's way
        # Create activity dataframe with values set by default to the transcription factor's expression
        activity = pd.DataFrame(self.design.loc[tfs, :].values,
                                index=tfs,
                                columns=self.design.columns)

        # #Debug
        # print "prior", self.priors_data.shape
        # print "expression_matrix", self.design.shape

        # Find all non-zero TFs that are duplicates of any other non-zero tfs
        is_duplicated = self.priors_data[non_zero_tfs].transpose().duplicated(keep=False)
        duplicates = is_duplicated[is_duplicated].index.tolist()

        # Find non-zero TFs that are also present in target gene list
        self_interacting_tfs = set(non_zero_tfs).intersection(self.priors_data.index)

        # If this flag is set to true, don't count duplicates as self-interacting when setting the diag to zero
        if allow_self_interactions_for_duplicate_prior_columns:
            self_interacting_tfs = self_interacting_tfs.difference(duplicates)

        # Set the diagonal of the matrix subset of self-interacting tfs to zero
        subset = self.priors_data.loc[self_interacting_tfs, self_interacting_tfs].values
        np.fill_diagonal(subset, 0)
        self.priors_data.at[self_interacting_tfs, self_interacting_tfs] = subset

        # Set the activity of non-zero tfs to the pseudoinverse of the prior matrix times the expression
        if non_zero_tfs:
            activity.loc[non_zero_tfs, :] = np.matrix(scipy.linalg.pinv2(self.priors_data[non_zero_tfs])) * np.matrix(
                self.half_tau_response)

        return activity  # , self.priors_data.copy()



