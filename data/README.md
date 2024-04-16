# Folders
## /data tools
**generate_psl.py** - Contains scripts for generating PSL logic accross multiple knowledge graphs.

**process_data.py** - Ingests predefined folds from the [OpenEA dataset](https://github.com/nju-websoft/OpenEA/) and outputs processed tsv files usable for training models.

You'll need to manually download the [OpenEA dataset](https://github.com/nju-websoft/OpenEA/) and configure the paths within to run these scripts.

## /psl50
Contains a preprocessed extract of the OpenEA EN/DE dataset with pre-generated PSL rules at a threshold of 50%
