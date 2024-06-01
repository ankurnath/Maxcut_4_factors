#!/bin/bash

# python evalute_generalization.py --train_distribution rnd_graph_800vertices_weighted --test_distribution rnd_graph_2000vertices_unweighted --model ECO_DQN
# python evalute_generalization.py --train_distribution rnd_graph_800vertices_weighted --test_distribution rnd_graph_2000vertices_weighted --model ECO_DQN
# python evalute_generalization.py --train_distribution rnd_graph_800vertices_weighted --test_distribution planar_2000vertices_weighted --model ECO_DQN
# python evalute_generalization.py --train_distribution rnd_graph_800vertices_weighted --test_distribution planar_2000vertices_unweighted --model ECO_DQN
# python evalute_generalization.py --train_distribution rnd_graph_800vertices_weighted --test_distribution toroidal_grid_2D_2000vertices_weighted --model ECO_DQN



# python evalute_generalization.py --train_distribution planar_800vertices_weighted --test_distribution rnd_graph_2000vertices_unweighted --model ECO_DQN
# python evalute_generalization.py --train_distribution planar_800vertices_weighted --test_distribution rnd_graph_2000vertices_weighted --model ECO_DQN
# python evalute_generalization.py --train_distribution planar_800vertices_weighted --test_distribution planar_2000vertices_weighted --model ECO_DQN
# python evalute_generalization.py --train_distribution planar_800vertices_weighted --test_distribution planar_2000vertices_unweighted --model ECO_DQN
# python evalute_generalization.py --train_distribution planar_800vertices_weighted --test_distribution toroidal_grid_2D_2000vertices_weighted --model ECO_DQN


python evalute_generalization.py --train_distribution toroidal_grid_2D_800vertices_weighted --test_distribution rnd_graph_2000vertices_unweighted --model ECO_DQN
python evalute_generalization.py --train_distribution toroidal_grid_2D_800vertices_weighted --test_distribution rnd_graph_2000vertices_weighted --model ECO_DQN
python evalute_generalization.py --train_distribution toroidal_grid_2D_800vertices_weighted --test_distribution planar_2000vertices_weighted --model ECO_DQN
python evalute_generalization.py --train_distribution toroidal_grid_2D_800vertices_weighted --test_distribution planar_2000vertices_unweighted --model ECO_DQN
python evalute_generalization.py --train_distribution toroidal_grid_2D_800vertices_weighted --test_distribution toroidal_grid_2D_2000vertices_weighted --model ECO_DQN