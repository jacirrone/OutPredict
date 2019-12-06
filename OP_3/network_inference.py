import numpy as np
import os
import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt



class NetworkInference:

    def __init__(self, flag_print, rnd_seed):
        self.flag_print = flag_print
        np.random.seed(rnd_seed)

    def calculate_precision_recall(self, combined_confidences, gold_standard):
        # this code only runs for a positive gold standard, so explicitly transform it using the absolute value:
        gold_standard = np.abs(gold_standard)
        # filter gold standard
        gold_standard_nozero = gold_standard.loc[(gold_standard!=0).any(axis=1), (gold_standard!=0).any(axis=0)]
        intersect_index = combined_confidences.index.intersection(gold_standard_nozero.index)
        intersect_cols = combined_confidences.columns.intersection(gold_standard_nozero.columns)
        gold_standard_filtered = gold_standard_nozero.loc[intersect_index, intersect_cols]
        combined_confidences_filtered = combined_confidences.loc[intersect_index, intersect_cols]
        # rank from highest to lowest confidence
        sorted_candidates = np.argsort(combined_confidences_filtered.values, axis = None)[::-1]
        gs_values = gold_standard_filtered.values.flatten()[sorted_candidates]

        if self.flag_print:
            print(("Num of intersected targets for AUPR computation: ", len(intersect_index)))
            print(("Num of intersected TFs for AUPR computation: ", len(intersect_cols)))
        num_edges_gs = np.sum(gs_values)

        if self.flag_print:
            print("Number of edges for AUPR computation: "+str(num_edges_gs))

        num_edges_gs = np.sum(gs_values)

        if self.flag_print:
            print(("Number of edges for AUPR computation: "+str(num_edges_gs)))

        random_aupr = float(num_edges_gs) / (float(len(intersect_index))*float(len(intersect_cols)))

        if self.flag_print:
            print("Random AUPR is: ", str(random_aupr))

        precision = np.cumsum(gs_values).astype(float) / np.cumsum([1] * len(gs_values))
        recall = np.cumsum(gs_values).astype(float) / sum(gs_values)
        precision = np.insert(precision,0,precision[0])
        recall = np.insert(recall,0,0)
        return (recall, precision, random_aupr)



    def calculate_aupr(self, recall, precision):
        #using midpoint integration to calculate the area under the curve
        d_recall = np.diff(recall)
        m_precision = precision[:-1] + np.diff(precision) / 2
        return sum(d_recall * m_precision)



    def plot_pr_curve(self, recall, precision, aupr, filename_prcurve):
        plt.figure()
        axes = plt.gca()
        axes.set_ylim([0,1])
        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.annotate("aupr = " + aupr.astype("str"), xy=(0.4, 0.05), xycoords='axes fraction')
        if self.flag_print:
            plt.savefig(filename_prcurve)#os.path.join(output_dir, 'pr_curve.pdf'))
        plt.close()



    def summarize_results(self, output_dir, filename_prcurve, combined_confidences, gold_standard, plot_flag):#, priors):

        (recall, precision, random_aupr) = self.calculate_precision_recall(combined_confidences, gold_standard)
        aupr = self.calculate_aupr(recall, precision)
        if plot_flag:
            filename_prcurve = os.path.join(output_dir, filename_prcurve)
            self.plot_pr_curve(recall, precision, aupr, filename_prcurve)
        return aupr, random_aupr
