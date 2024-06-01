# import subprocess
import os
import numpy as np

# # Python program to show time by perf_counter() 
# from time import perf_counter
# import pandas as pd
# t1_start = perf_counter() 
# # command = 'python LocalSearch/train.py --distribution ER_20 --model ECO_DQN'

# command = 'python LocalSearch/evaluate.py --distribution ER_20 --model LinearRegression'
# command = 'python LocalSearch/evaluate.py --distribution BA_20 --model LinearRegression'

# command = 'python LocalSearch/evaluate_heurestics.py --distribution ER_20'
# command = 'python LocalSearch/CIM.py --distribution ER_20'

# command = 'python ANYCSP/evaluate.py --distribution BA_200'
# command = 'python ecord/evaluate.py --distribution ER_20'
# # command = 'python gflow/evaluate.py --distribution Physics'
# command = 'python gflow/evaluate.py --distribution Physics'
# command = 'python RUNCSP/evaluate.py --distribution Physics'
# t1_stop = perf_counter()
# gpu_commmand='nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv -l 1 -f gpu_log.csv'

# subprocess.Popopen(gpu_commmand, shell=True, check=True)




# subprocess.run(command, shell=True, check=True)

# print("Elapsed time during the whole program in seconds:",
#                                         t1_stop-t1_start)

# data=pd.read_csv('gpu_log.csv')
# gpu_usage= data[data[' process_name']=='python']

# print('Mean GPU usage:',gpu_usage[' used_gpu_memory [MiB]'].mean())
# for process_name,group_df in data.groupby(' process_name'):
#     print(process_name)


import subprocess

# subprocess.run(f'python LocalSearch/evaluate.py --distribution rnd_graph_800vertices_weighted --model ECO_DQN', shell=True, check=True)
# # subprocess.run(f'python LocalSearch/evaluate.py --distribution rnd_graph_800vertices_unweighted --model ECO_DQN', shell=True, check=True)
# subprocess.run(f'python LocalSearch/evaluate.py --distribution planar_800vertices_unweighted --model ECO_DQN', shell=True, check=True)
# subprocess.run(f'python LocalSearch/evaluate.py --distribution planar_800vertices_weighted --model ECO_DQN', shell=True, check=True)
# subprocess.run(f'python LocalSearch/evaluate.py --distribution toroidal_grid_2D_800vertices_weighted --model ECO_DQN', shell=True, check=True)


# subprocess.run(f'python LocalSearch/evaluate.py --distribution rnd_graph_800vertices_weighted --model LinearRegression', shell=True, check=True)
# # subprocess.run(f'python LocalSearch/evaluate.py --distribution rnd_graph_800vertices_unweighted --model LinearRegression', shell=True, check=True)
# subprocess.run(f'python LocalSearch/evaluate.py --distribution planar_800vertices_unweighted --model LinearRegression', shell=True, check=True)
# subprocess.run(f'python LocalSearch/evaluate.py --distribution planar_800vertices_weighted --model LinearRegression', shell=True, check=True)
# subprocess.run(f'python LocalSearch/evaluate.py --distribution toroidal_grid_2D_800vertices_weighted --model LinearRegression', shell=True, check=True)


# subprocess.run(f'python LocalSearch/evaluate.py --distribution rnd_graph_800vertices_weighted --model LSDQN', shell=True, check=True)
# # subprocess.run(f'python LocalSearch/evaluate.py --distribution rnd_graph_800vertices_unweighted --model LSDQN', shell=True, check=True)
# subprocess.run(f'python LocalSearch/evaluate.py --distribution planar_800vertices_unweighted --model LSDQN', shell=True, check=True)
# subprocess.run(f'python LocalSearch/evaluate.py --distribution planar_800vertices_weighted --model LSDQN', shell=True, check=True)
# subprocess.run(f'python LocalSearch/evaluate.py --distribution toroidal_grid_2D_800vertices_weighted --model LSDQN', shell=True, check=True)




from time import perf_counter
import pandas as pd

algorithm='S2V'
distribution='rnd_graph_800vertices_unweighted'

# Define the command to run

if algorithm in ['ECO_DQN','LSDQN','S2V','LinearRegression',]:
    command = f'python LocalSearch/evaluate.py --distribution {distribution} --model {algorithm}'
elif algorithm in ['ANYCSP']:
    command = f'python ANYCSP/evaluate.py --distribution {distribution} --network_steps 1600'

elif algorithm in ['ECORD']:
    command = f'python ecord/evaluate.py --distribution {distribution}'

elif algorithm in ['RUNCSP']:
    command = f'python RUNCSP/evaluate.py --distribution {distribution} --network_steps 1600'

elif algorithm in ['Gflow-CombOpt']:
    command = f'python gflow/evaluate.py --distribution {distribution} '



else:

    pass



# Command to log GPU usage
gpu_command = f'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv -l 1 -f gpu_log_{distribution}_{algorithm}.csv'

print(gpu_command)
# # Start logging GPU usage
gpu_process = subprocess.Popen(gpu_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# # Wait for a short time to ensure logging has started
# # Adjust this as needed based on how quickly logging starts
# # You may need to increase the wait time or use a more reliable method to ensure logging starts properly.
import time
time.sleep(20)

# Start measuring elapsed time
t1_start = perf_counter()

# # Run the main command
subprocess.run(command, shell=True, check=True)
#  Measure elapsed time after command completion
t1_stop = perf_counter()
time.sleep(20)

# # Stop logging GPU usage
gpu_process.terminate()

# Wait for the process to terminate
try:
    gpu_process.wait(timeout=20)
except subprocess.TimeoutExpired:
    gpu_process.kill()







time.sleep(60)
try:
# # Read GPU log data
    data = pd.read_csv(f'gpu_log_{distribution}_{algorithm}.csv')
    data = data.dropna()
    
    os.remove(f'gpu_log_{distribution}_{algorithm}.csv')
    column_names = data.columns
    new_column_names = [name.strip() for name in column_names]
    data.columns = new_column_names
    # print(data)

    # Calculate GPU usage statistics
    gpu_usage = data[data['process_name'] == ' python']
    # print(gpu_usage)
    mean_gpu_usage = gpu_usage['used_gpu_memory [MiB]'].str.replace(' MiB', '').astype(int).mean()
    # print(mean_gpu_usage)
except:
    mean_gpu_usage=None

# Print results
print("Elapsed time during the whole program in seconds:", t1_stop - t1_start)
print('Mean GPU usage:', mean_gpu_usage)

folder='Benchmark'
os.makedirs(folder,exist_ok=True)

benchmark_data={'Distribution':[distribution],'Algorthim':[algorithm],'Mean GPU Usage':[mean_gpu_usage],'Elapsed Time (s)':[t1_stop - t1_start]}
benchmark_data=pd.DataFrame(benchmark_data)
print(benchmark_data)

file_path=os.path.join(folder,f'{distribution}_{algorithm}')
benchmark_data.to_pickle(file_path)

