import re
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import glob
print(sns.__version__)
legend_list = []

def plot_each_folder(path,color="red"):
    evaluation_csv = glob.glob(path  + "/**/accs/her_result1/*.npz", recursive=True)
    colors = sns.color_palette("Set2", len(evaluation_csv))
    pattern = r'\d{14}_\w{5}FetchPush'
    # train_csv = glob.glob(path  + "/**/train.csv", recursive=True)
    labels=[]
    for k, eval_file in enumerate(evaluation_csv):
        x_data = []
        y_data = []
        match = re.search(pattern, eval_file)

        df = np.load(eval_file)['info']
        epoch_number = df[:,0]
        success_result_array = df[:,1]
        x_data.append(epoch_number)
        y_data.append(success_result_array)
        print(eval_file)

        sns.tsplot(
                time=x_data[0],
                data=y_data,
                color=colors[k],
                linestyle="-"
            )

        labels.append(match.group())
    plt.legend(labels=labels,loc='lower left', )
    return x_data,y_data


def plot_evaluation_graph(path,color="red"):
    evaluation_csv = glob.glob(path  + "*.npz", recursive=True)

    # train_csv = glob.glob(path  + "/**/train.csv", recursive=True)
    x_data = []
    y_data = []
    for eval_file in evaluation_csv:

        df = np.load(eval_file)['info']
        epoch_number = df[:,0]
        success_result_array = df[:,1]
        x_data.append(epoch_number)
        y_data.append(success_result_array)


    sns.tsplot(
            time=x_data[0],
            data=y_data,
            color=color,
            linestyle="-"
        )
    
    return x_data,y_data








path = "/media/"

x_data, y_data = plot_evaluation_graph(path, 'red')

path = "/media/"

x_data, y_data = plot_evaluation_graph(path, 'green')

# sns.tsplot(time=x_data[0], data=np.ones(len(x_data[0])), color="black", linestyle="-")


plt.title("FetchPickAndPlace", fontsize=15)
plt.ylabel("Success Rate", fontsize=15)
plt.xlabel("Episode ", fontsize=15, labelpad=4)
plt.legend(labels=['hgg','diffusion'],loc='lower left', )
# plt.savefig('FetchReach_interval.pdf') 
# plt.legend(labels=legend_list)
plt.show()

# logs.close()
