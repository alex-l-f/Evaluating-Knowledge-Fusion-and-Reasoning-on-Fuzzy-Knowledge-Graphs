import os
import glob
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

def extract_epoch_data(directory):
    epoch_data = []
    folders = [folder for folder in os.listdir(directory) if folder.isdigit()]

    for folder in folders:
        tsv_files = glob.glob(os.path.join(directory, folder, '*.tsv'))
        epoch_numbers = []

        for tsv_file in tsv_files:
            with open(tsv_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    epoch, _, _ = line.split('\t')
                    epoch_numbers.append(int(epoch))

        epoch_numbers.sort()
        epoch_data.append(epoch_numbers[-30])

    return epoch_data

def compute_average_error(epoch_data):
    average_epoch = np.mean(epoch_data)
    #95% confidence interval
    error = np.std(epoch_data)*1.96

    return average_epoch, error

#add log directories and names here
directories = ['./results/models/psl50/EN_DE_100K_V2']
names = ['PSL 50%']

#collect data from each directory, then plot to compare
data = {}
for directory in directories:
    epoch_data = extract_epoch_data(directory)
    average_epoch, error = compute_average_error(epoch_data)
    data[directory] = (average_epoch, error)

#plot the epochs together
plt.figure()
plt.bar(names, [data[directory][0] for directory in directories], yerr=[data[directory][1] for directory in directories], capsize=5)
plt.ylabel('Epoch')
#lt.title('Average Epochs for EN_DE dataset')

plt.savefig('epochs.png', dpi=100)
