import os
from pandas import DataFrame
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def get_logs()->dict:
    logs = defaultdict(dict)
    for f in os.listdir():
        if f.startswith('Net'):
            nt, conv = f.split('-')
            
            logs[nt][conv] = []

            with open(f, 'r') as lg:
                for line in lg:
                    if not line.startswith('*') and not line.startswith('Type'):
                        stripped = line.strip('\n')
                        logs[nt][conv].append(stripped)
    return logs

def get_dataframe(logs:dict)->DataFrame:
    df = {'Net':[],
          'Conv':[],
          'Accuracy':[],
          'SD':[],
          'Dataset':[]}
    for net, net_dict in logs.items():
        for conv, conv_log in net_dict.items():
            for res in conv_log:
                if res.startswith('Results'):
                    continue
                df['Net'].append(net)
                df['Conv'].append(conv)
                res = res.split(' ')
                dataset = res[-1]
                df['Dataset'].append(dataset)
                df['Accuracy'].append(float(res[1]))
                df['SD'].append(float(res[3]))
    return DataFrame(df)

def plot_results(df:DataFrame):
    sns.lineplot(df, x = 'Conv', y = 'Accuracy', hue = 'Net', style = 'Dataset'
                 )
    plt.tight_layout(); plt.show()
if __name__ == '__main__':
    logs = get_logs()
    df = get_dataframe(logs)
    print(df)
    plot_results(df)