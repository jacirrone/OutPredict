% Copyright (C) 2019 Jacopo Cirrone


This repository contains OutPredict, a python developed Method for Predicting Out-of-sample Data in Time Series and Steady State data as well as to predict
Causal connections from transcription factors to genes. 

Here we consider the problem of predicting the expression of a large set of genes at a future time point given the previous time points in a genomic time series. The prediction may rest on several sources of data: the time point immediately preceding t, the entire target time series preceding t, and ancillary data. In biology, for example, the ancillary data is dependent on the myriad experimental design choices and may consist of a network based on binding data, data from different time series, steady state data, a community-blessed gold standard network, or some combination of those. OutPredict is a machine learning method for time series that incorporates ancillary steady state and network data to achieve a low error on gene expression prediction. It outperforms several of the best state-of-the-art methods in network inference.


In addition to time-series gene expression forecasting, OutPredict supports the prediction  of leave-out steady-state expression data even when no time series data is available.

## Installation for OutPredict

To run OutPredict, the latest version of Miniconda or Anaconda must be previously installed (Anaconda: https://www.anaconda.com/distribution/#download-section)

If conda is already installed on your machine, you can update to the latest version with:

```
conda update --all
```

Clone the codebase:

```
git clone https://github.com/jacirrone/OutPredict.git
```

Enter the OP_3 directory:

```
cd OP_3/
```

To install OutPredict, first install the OpenMP library as follows:

For Mac-OS:
```
brew install libomp
```

For Linux:

(Side note: it might be necessary to run "sudo apt-get update" and "sudo apt-get install gcc", especially if you are using a virgin AWS machine, for example)

```
sudo apt-get install libomp-dev clang
```

Then, run (in the OP3/ directory) the OutPredict Installation file:

```
sh install.sh
```


As example to run OutPredict, invoke the corresponding pipeline script for the dream10 dataset:
 
python dream10_pipeline.py








## Required data for OutPredict

The Datasets directory, "OP_3/Datasets/", contains the directories of each organism's datasets.

Let us consider a dataset for a generic organism called "new_organism".
Inside this directory "OP_3/Datasets/new_organism/" the following sample files are required 
(for the "dream10" example the directory is "OP_3/Datasets/dream10/"):

expression.tsv
-----------------
expression values; must include row (genes) and column (conditions) names

Obtain expression data and save it as a tsv file "expression.tsv" of [Genes x Samples]


gold_standard.tsv (required when choosing the gold standard priors option on OutPredict, see below)
-----------------
needed for OutPredict with "gold_standard" priors; matrix of 0s and 1s indicating whether we have prior knowledge about 
the interaction of a transcription factor (TF) and a gene; one row for each gene, one column for each TF; must include row (genes) and column (TF) names.

So the position tf t and gene g: is 1 if there is an inductive or repressive edge; is 0 if there is no such edge or unknown.

Obtain gold standard data, interactions between TFs and target genes and save it as a tsv file "gold_standard.tsv" [Genes x TFs]

interaction_weights_list.tsv (required when choosing the steady state priors option on OutPredict, see below)
-----------------
needed for OutPredict with "steady_state" priors; 
This type of priors is represented by a list of interactions indicating whether we have prior knowledge about 
the interaction of a transcription factor (TF) and a gene. In this case the prior knowledge is represented by a real number weight, which is an interaction confidence score.
First column are TFs, second column are genes, third column are real number weights.


meta_data.tsv
-------------
In a gene expression dataset a condition is defined as an experimental assay or a replicate of an experiment.

The meta data file describes the conditions; must include column names;
has five columns:

isTs: TRUE if the condition is part of a time-series, FALSE else.

is1stLast: "e" if the condition is
not part of a time-series; "f" if first; "m" middle; "l" last.
Thus "l" means a value at the last time point, "f" at the first time point, "m" all 
others.

prevCol: name of the preceding condition in time-series; NA if "e" or "f".

del.t: time in whatever the common unit (e.g. typically minutes
for transcription factors) since prevCol; NA if "e" or "f".

condName: name of the condition.


tf_names.tsv
------------
one TF (transcription factor)
 name on each line; these must be subset of the row names of the expression data

Create a list of TFs to model for inference and save it as a file "tf_names.tsv" with each TF on a separate line [TFs]

Note that each gene 
(TFs and others)  
must have the same name
in all files (expression, gold_standard, etc.)









## Construct a new run script (`pipeline_new_organism.py`) for a generic organism
## Here is an example of the contents of that file:






Create an OutPredict instance:
```
from OutPredict import *

if __name__ == '__main__':
    op = OutPredict()

```
Set required file names and parameters:
--------------------------------------------------------------------------------
```


#[default is 300]
op.num_of_trees = 300 # The number of Trees for Random Forests

op.input_dir_name = "dream10"  # Name of Directory, inside OP_3/Datasets/, containing the dataset

#[default is 0.15]
op.test_set_split_ratio = 0.15  # The percentage of data points to use for the test set separately for time-series and steady-state, e.g. 0.15, 15% of steady-state data will be used as test set, 15% of the time-series data (last time points of time-series)

#[default is TS-SS]
op.training_data_type = "TS-SS"  # whether to use for training TS(time-series), SS(steady-stae) or TS-SS (time-series and steady-state)

#[default is TS]
op.leave_out_data_type = "TS"  # whether to use for training TS(time-series), SS(steady-stae) or TS-SS (time-series and steady-state)

#[default is 0]
op.genes_coeff_of_var_threshold = 0  # coefficient of variance threshold to filter the genes to modeling; 0 to modeling all genes

#[default is 20]
op.num_of_cores = 20  # (Integer) number of cores to use for parallelization

#it's not necessary to set which method to use - either time-step or ode-log - because it will be automatically learned


Set required params to run OutPredict WITH Priors:
--------------------------------------------------------------------------------

op.prior_file_name = "gold_standard.tsv"  # either name of file containing prior knowledge or the empty string if there are no priors.

op.priors = "gold_standard"  # gold_standard or steady_state or the empty string if there is no gold standard data


op.run()

```








## Run OutPredict

Enter the OP_3 directory and activate the conda environment op3

```
cd OP_3/
conda activate op3
```

To use OutPredict WITHOUT priors, do NOT set the params "prior_file_name" and "priors", and this script can now be run from the command line as 
```
python -s pipeline_new_organism.py

```


To use OutPredict WITH priors, after properly setting BOTH the params "prior_file_name" and "priors", this script can now be run from the command line as 
```
python pipeline_new_organism.py

```




The folder "OP_3/output/" contains the output folders for the different runs of OutPredict.

A generic output folder for a run related to the "WITHOUT priors" version is called 
"new_organism_output_RF_...".

A generic output folder for a run related to the "WITH priors" version is called 
"new_organism_output_RF-mod_..."