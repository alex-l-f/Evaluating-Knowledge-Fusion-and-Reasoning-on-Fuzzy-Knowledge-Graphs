Repository for my CMPUT 656 Project

# Overview
Almost everthing is preconfigured to run as is. Exceptions are noted in the folder specific readmes. 
Configuration is not argument based, but requires editing configuration lines at the beginning of files.

All code is expected to be run with from the root project directory. I recommend using visual studio code or a similar IDE to do so automatically.

A requirements.txt file is included, thought the only hard requirements should be pytorch, pyplotlib, tqdm, and numpy. 
Specific version of numpy can give errors when generating the PCA plots, so if you encounter those I recommened using the version specified in the requirements.

Finally, this code has only been tested on PCs with CUDA compatible GPUs, though the relevant scripts should easily be changeable by altering the device line in the config block.
It's untested though, and probably too slow to be useful anyways.

# Folders
Some folders will contain a readme describing the contents further.

## /UKGE 
Reimplementation of the [UKGE paper](https://arxiv.org/abs/1811.10667), along with scripts to run single trials or groups of trials.

## /data
Contains the scripts used to process the OpenEA dataset and used to generate PSL logic, along with one example of preprocessed data for the PSL 50% trials.

## /evaluate
A set of scripts to evaluate trained models and my PRA implementation, and also generate the videos, figures and tables used in the final paper and presentation.

## /results
All the tables, figures, videos, and misc data output at various stages of a full run.
Includes the models for two folds of the PSL 50% trials (others left out due to repository space contraints).
