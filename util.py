import os
import subprocess
from tabulate import tabulate
import re
import pandas as pd
import numpy as np
import select
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np

def get_paths(name,args=('1','2','3')):
    os.makedirs(os.path.join(os.getcwd(), name),exist_ok=True)
    dir = [os.getcwd(),os.path.join(os.getcwd(),name)]
    for arg in args:
        os.makedirs(os.path.join(os.getcwd(),name,arg),exist_ok=True)
        dir.append(os.path.join(os.getcwd(),name,arg))
    return tuple(dir)
# def run_commands(commands, log=False):
#     output, error, output_list = ([], [], [])
#     for i, command in enumerate(commands):
#         if type(command) == str:
#             process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         if type(command) == list:
#             process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         stdout, stderr = process.communicate()
#         output.append(stdout.decode())
#         error.append(stderr.decode())
#         if log == True:
#             output_list.append([str(i + 1), command, stdout.decode(), stderr.decode()])
#     headers = ["#", "Command", "Output_report", "Error_report"]
#     if log == True:
#         with open('output.txt', 'a') as f:
#             f.write(tabulate(output_list, headers, tablefmt="fancy_grid"))
def output_parser(file_name: str, error_file: str, output_types: list) -> pd.DataFrame:
        output_values = {output_type: 0 for output_type in output_types}
        if os.path.isfile(error_file):
            df = pd.DataFrame(output_values, index=["0"])
            return df
        else:
            with open(file_name, 'r') as f:
                lines = f.read()
                output_types_re = "|".join(output_types)
                matches = re.finditer(output_types_re, lines)
                for match in matches:
                    output_type = match.group()
                    match = re.search(r'-?\d+\.\d+', lines[match.start():])
                    if match:
                        output_values[output_type] = match.group()
            df = pd.DataFrame(output_values, index=["0"])
            return df
def xyz_generator(num:str,contents:np.array,name:str='coord.xyz'):
    with open(name, 'w') as f:
        f.write(num)
        f.write('\n\n')
        np.savetxt(f, contents , fmt='%s')


def merge_grid_eda(energy_file, xyz_file, cavity_file):
    if 'xTB' in energy_file:
        a = pd.read_csv(energy_file, sep='\s+', header=None, skiprows=2, usecols=range(0, 9))
    elif 'Turbomole' in energy_file:
        a = pd.read_csv(energy_file, sep='\s+', header=None, skiprows=2, usecols=range(1, 11))

    b = pd.read_csv(xyz_file, sep='\s+', header=None, skiprows=2, usecols=range(0, 4))
    cavity = pd.read_csv(cavity_file, sep='\s+', header=None, skiprows=2, usecols=range(0, 4))
    c = np.column_stack((b.to_numpy(), a.to_numpy()))
    d = np.column_stack((cavity.to_numpy(), np.full((len(cavity), a.to_numpy().shape[1]), 2)))
    e = np.vstack((c, d))
    with open(f'{xyz_file.split("_")[0]}_total.xyz', 'w') as f:
        f.write(e.shape[0].__str__())
        f.write('\n\n')
        np.savetxt(f, e, fmt='%s')


def run_commands(commands, timeout=0, log=False, output_file="output.txt", ):
    output_list = []
    process = subprocess.Popen(["bash"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
    for i, command in enumerate(commands):
        process.stdin.write(f"{command}\n")
        process.stdin.flush()
        rlist, _, _ = select.select([process.stdout, process.stderr], [], [], timeout)
        if log is True:
            for r in rlist:
                if r is process.stdout:
                    print(r)
                    output = process.stdout.readline()
                    output_list.append(["Command " + str(i + 1), command, "stdout", output])
                else:
                    error = process.stderr.readline()
                    output_list.append(["Command " + str(i + 1), command, "stderr", error])
    stdout, stderr = process.communicate()
    headers = ["Command Number", "Command", "Output Type", "Output"]
    if log is True:
        f = open(output_file, "w")
        f.write(tabulate(output_list, headers, tablefmt="fancy_grid"))
        f.close()

def MIF_filter(clusters=10,threshold=0.2,gridpoints:np.array=None):
    kmeans = KMeans(clusters)
    kmeans.fit(gridpoints)
    labels = kmeans.labels_
    filtered = []
    for i in range(0,clusters,1):
        new = gridpoints[labels==i]
        distances = cdist(gridpoints[labels==i],gridpoints[labels==i])
        mask = np.ones(len(gridpoints[labels==i]), dtype=bool)
        for j in range(len(gridpoints[labels==i])):
            if mask[j]:
                mask[(j+1):][distances[j, (j+1):] < threshold] = False
        filtered.append(gridpoints[labels==i][mask])
    filtered = np.concatenate(filtered,axis=0)
    return filtered

def MIF_filter_recursion(clusters=10,gridpoints:np.array=None):
    def MIF_filter(clusters=10,gridpoints:np.array=None):
        kmeans = KMeans(clusters)
        kmeans.fit(gridpoints)
        labels = kmeans.labels_
        filtered = []
        for i in range(0,clusters,1):
            new = gridpoints[labels==i]
            #every three rows remove one row
            new = np.delete(new,np.arange(0,new.shape[0],10),axis=0)
            filtered.append(new)
        filtered = np.concatenate(filtered,axis=0)
        return filtered
    filtered = MIF_filter(clusters,gridpoints)
    if filtered.shape[0] < 1200:
        return filtered
    else:
        return MIF_filter_recursion(clusters+1,filtered)