import os
import subprocess
from tabulate import tabulate
import re
import pandas as pd
import numpy as np

def get_paths(name,args=('1','2','3')):
    os.makedirs(os.path.join(os.getcwd(), name),exist_ok=True)
    dir = [os.getcwd(),os.path.join(os.getcwd(),name)]
    for arg in args:
        os.makedirs(os.path.join(os.getcwd(),name,arg),exist_ok=True)
        dir.append(os.path.join(os.getcwd(),name,arg))
    return tuple(dir)
def run_commands(commands, log=False):
    output, error, output_list = ([], [], [])
    for i, command in enumerate(commands):
        if type(command) == str:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if type(command) == list:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        output.append(stdout.decode())
        error.append(stderr.decode())
        if log == True:
            output_list.append([str(i + 1), command, stdout.decode(), stderr.decode()])
    headers = ["#", "Command", "Output_report", "Error_report"]
    if log == True:
        with open('output.txt', 'a') as f:
            f.write(tabulate(output_list, headers, tablefmt="fancy_grid"))
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