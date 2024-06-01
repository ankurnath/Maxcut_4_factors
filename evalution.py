import subprocess



for distribution in ['rnd_graph_800vertices_unweighted',
                     'rnd_graph_800vertices_weighted',
                     'planar_800vertices_unweighted',
                     'planar_800vertices_weighted',
                     'toroidal_grid_2D_800vertices_weighted']:
    for algorithm in ['ECO_DQN','LSDQN','RUNCSP','ANYCSP','ECROD-det']:

        if algorithm in ['ECO_DQN','LSDQN','S2V','LinearRegression',]:
            command = f'python LocalSearch/evaluate.py --distribution {distribution} --model {algorithm} --step_factor 4'
        elif algorithm in ['ANYCSP']:
            command = f'python ANYCSP/evaluate.py --distribution {distribution} --network_steps 3200'

        elif algorithm in ['ECORD']:
            command = f'python ecord/evaluate.py --distribution {distribution} --step_factor -4'
        
        elif algorithm in ['ECROD-det']:
            command = f'python ecord/evaluate.py --distribution {distribution} --step_factor -4 --tau 0'

        elif algorithm in ['RUNCSP']:
            command = f'python RUNCSP/evaluate.py --distribution {distribution} --network_steps 3200'

        subprocess.run(command, shell=True, check=True)


for distribution in ['ER_200',
                     'BA_200',
                     'HomleKim_200vertices_unweighted',
                     'HomleKim_200vertices_weighted',
                     'WattsStrogatz_200vertices_unweighted',
                     'WattsStrogatz_200vertices_weighted',
                     'dense_MC_100_200vertices_unweighted']:
    for algorithm in ['ECO_DQN','LSDQN','RUNCSP','ANYCSP','ECROD-det']:

        if algorithm in ['ECO_DQN','LSDQN','S2V','LinearRegression',]:
            command = f'python LocalSearch/evaluate.py --distribution {distribution} --model {algorithm} --step_factor 4'
        elif algorithm in ['ANYCSP']:
            command = f'python ANYCSP/evaluate.py --distribution {distribution} --network_steps 800'

        elif algorithm in ['ECORD']:
            command = f'python ecord/evaluate.py --distribution {distribution} --step_factor -4'
        
        elif algorithm in ['ECROD-det']:
            command = f'python ecord/evaluate.py --distribution {distribution} --step_factor -4 --tau 0'

        elif algorithm in ['RUNCSP']:
            command = f'python RUNCSP/evaluate.py --distribution {distribution} --network_steps 800'

        subprocess.run(command, shell=True, check=True)


for distribution in ['Physics',
                     ]:
    for algorithm in ['ECO_DQN','LSDQN','RUNCSP','ANYCSP','ECROD-det']:

        if algorithm in ['ECO_DQN','LSDQN','S2V','LinearRegression',]:
            command = f'python LocalSearch/evaluate.py --distribution {distribution} --model {algorithm} --step_factor 4'
        elif algorithm in ['ANYCSP']:
            command = f'python ANYCSP/evaluate.py --distribution {distribution} --network_steps 500'

        elif algorithm in ['ECORD']:
            command = f'python ecord/evaluate.py --distribution {distribution} --step_factor -4'
        
        elif algorithm in ['ECROD-det']:
            command = f'python ecord/evaluate.py --distribution {distribution} --step_factor -4 --tau 0'

        elif algorithm in ['RUNCSP']:
            command = f'python RUNCSP/evaluate.py --distribution {distribution} --network_steps 500'

        subprocess.run(command, shell=True, check=True)

for distribution in ['Physics',
                     ]:
    for algorithm in ['ECO_DQN','LSDQN','RUNCSP','ANYCSP','ECROD-det']:

        if algorithm in ['ECO_DQN','LSDQN','S2V','LinearRegression',]:
            command = f'python LocalSearch/evaluate.py --distribution {distribution} --model {algorithm} --step_factor 4'
        elif algorithm in ['ANYCSP']:
            command = f'python ANYCSP/evaluate.py --distribution {distribution} --network_steps 500'

        elif algorithm in ['ECORD']:
            command = f'python ecord/evaluate.py --distribution {distribution} --step_factor -4'
        
        elif algorithm in ['ECROD-det']:
            command = f'python ecord/evaluate.py --distribution {distribution} --step_factor -4 --tau 0'

        elif algorithm in ['RUNCSP']:
            command = f'python RUNCSP/evaluate.py --distribution {distribution} --network_steps 500'

        subprocess.run(command, shell=True, check=True)


for distribution in ['SK_spin_70_100vertices_weighted',
                     ]:
    for algorithm in ['ECO_DQN','LSDQN','RUNCSP','ANYCSP','ECROD-det']:

        if algorithm in ['ECO_DQN','LSDQN','S2V','LinearRegression',]:
            command = f'python LocalSearch/evaluate.py --distribution {distribution} --model {algorithm} --step_factor 4'
        elif algorithm in ['ANYCSP']:
            command = f'python ANYCSP/evaluate.py --distribution {distribution} --network_steps 400'

        elif algorithm in ['ECORD']:
            command = f'python ecord/evaluate.py --distribution {distribution} --step_factor -4'
        
        elif algorithm in ['ECROD-det']:
            command = f'python ecord/evaluate.py --distribution {distribution} --step_factor -4 --tau 0'

        elif algorithm in ['RUNCSP']:
            command = f'python RUNCSP/evaluate.py --distribution {distribution} --network_steps 400'

        subprocess.run(command, shell=True, check=True)
