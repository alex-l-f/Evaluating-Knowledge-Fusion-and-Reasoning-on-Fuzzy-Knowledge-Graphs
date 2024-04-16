import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 80})

# Initialize empty lists to store the data
datasets = []
mses = []
maes = []
h1s = []
h5s = []
h10s = []
h50s = []
mrs = []
mrrs = []

# Read the results.tsv file
with open('./results/model_eval_results.tsv', 'r') as file:
    for line in file:
        #skip the header
        if "{data_set}" in line:
            continue

        # Split the line by tab
        data = line.strip().split('\t')
        
        # Extract the values
        dataset = data[0].split('/')[0]
        mse = float(data[1])
        mae = float(data[2])
        h1 = float(data[3])
        h5 = float(data[4])
        h10 = float(data[5])
        h50 = float(data[6])
        mr = float(data[7])
        mrr = float(data[8])
        
        # Append the values to the respective lists
        datasets.append(dataset)
        mses.append(mse)
        maes.append(mae)
        h1s.append(h1)
        h5s.append(h5)
        h10s.append(h10)
        h50s.append(h50)
        mrs.append(mr)
        mrrs.append(mrr)

#compute means and 95% error intervals for each dataset and metric

#split the data by dataset
data = {}
for i in range(len(datasets)):
    if datasets[i] not in data:
        data[datasets[i]] = {}
        data[datasets[i]]["mse"] = []
        data[datasets[i]]["mae"] = []
        data[datasets[i]]["hits@1"] = []
        data[datasets[i]]["hits@5"] = []
        data[datasets[i]]["hits@10"] = []
        data[datasets[i]]["hits@50"] = []
        data[datasets[i]]["mr"] = []
        data[datasets[i]]["mrr"] = []
    data[datasets[i]]["mse"].append(mses[i])
    data[datasets[i]]["mae"].append(maes[i])
    data[datasets[i]]["hits@1"].append(h1s[i])
    data[datasets[i]]["hits@5"].append(h5s[i])
    data[datasets[i]]["hits@10"].append(h10s[i])
    data[datasets[i]]["hits@50"].append(h50s[i])
    data[datasets[i]]["mr"].append(mrs[i])
    data[datasets[i]]["mrr"].append(mrrs[i])
        

#compute the means and 95% error intervals
means = {}
errors = {}
for dataset in data:
    means[dataset] = {}
    errors[dataset] = {}
    for metric in data[dataset]:
        means[dataset][metric] = np.mean(data[dataset][metric])
        errors[dataset][metric] = 1.96*np.std(data[dataset][metric])/np.sqrt(len(data[dataset][metric]))

# Define the metrics and colors
metrics = ["mse", "mae", "mr", "mrr", "hits@1", "hits@5", "hits@10", "hits@50"]

# Create subplots for each metric
fig, axs = plt.subplots(2, 4, figsize=(90, 40), sharex=True)

# Plot the data for each metric
for i, metric in enumerate(metrics):
    ax = axs[i//4, i%4]
    ax.set_title(metric.upper())
    
    # Plot the data for each dataset
    mindex = -1
    for j, dataset in enumerate(data):
        #get index of dataset with lowest mean value
        if mindex == -1 or means[dataset][metric] < means[mindex][metric]:
            mindex = dataset
        ax.bar(j, means[dataset][metric], yerr=errors[dataset][metric], capsize=30)
    #clip the lower bounds of the yaxis to near lowest value - error
    val = means[mindex][metric] - errors[mindex][metric]
    ax.set_ylim(bottom=val*0.9)

    ax.set_xticks([])
    #ax.set_xticks(range(len(data)))
    #ax.set_xticklabels(data.keys())

plt.tight_layout()

#set legend below plot
fig.subplots_adjust(bottom=0.1)   ##  Need to play with this number.
fig.legend(labels=set(datasets) ,loc="lower center", ncol=4)

#save the plot
plt.savefig('evals_plot.png', dpi=100)

#write out latex table of means and errors
#be sure to write the error intervals in scientific notation
with open('results_table.tex', 'w') as f:
    f.write("\\begin{tabular}{c|cccc}\n")
    f.write("Dataset & MSE & MAE & MR & MRR\\\\\n")
    f.write("\\hline\n")
    for dataset in data:
        f.write(f"{dataset}".replace("%", "\\%"))
        for metric in metrics:
            if "hits" in metric:
                continue
            f.write(f" & {means[dataset][metric]:.2e} $\pm$ {errors[dataset][metric]:.2e}")
        f.write(" \\\\\n")
    f.write("\\end{tabular}")
    f.write("\\begin{tabular}{c|cccc}\n")
    f.write("Dataset & Hits@1 & Hits@5 & Hits@10 & Hits@50\\\\\n")
    f.write("\\hline\n")
    for dataset in data:
        f.write(f"{dataset}".replace("%", "\\%"))
        for metric in metrics:
            if "hits" not in metric:
                continue
            f.write(f" & {means[dataset][metric]:.2e} $\pm$ {errors[dataset][metric]:.2e}")
        f.write(" \\\\\n")
    f.write("\\end{tabular}")