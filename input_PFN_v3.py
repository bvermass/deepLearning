#name of input root file
root_file_name = '/user/bvermass/public/2l2q_analysis/trees/HNLtagger/final/full_analyzer/HNLtagger_muon_randomized.root'

validation_fraction = 0.4
test_fraction = 0.2

number_of_threads = 4

test = False

if test:
    parameter_values = {
        'hidden_layers_latent' : [2],
        'nodes_latent' : [128],
        'dropout_latent' : [0.1],
        'latent_space' : [256],
        'hidden_layers_output' : [2],
        'nodes_output' : [128],
        'dropout_output' : [0.1],
        'learning_rate' : [1],
        'learning_rate_decay' : [1]
    }

else:
    parameter_values = {
        'hidden_layers_latent' : [3, 4, 5],
        'nodes_latent' : [128, 256],
        'dropout_latent' : [0.2, 0.3, 0.5],
        'latent_space' : [256, 512],
        'hidden_layers_output' : [3, 4, 5],
        'nodes_output' : [128, 256],
        'dropout_output' : [0.2, 0.3, 0.5],
        'learning_rate' : [0.1, 1],
        'learning_rate_decay' : [1]
    }

