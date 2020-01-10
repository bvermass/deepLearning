#name of input root file
root_file_name = '/user/bvermass/public/PFN/JetTagger/PFN_v5/HNLtagger_muon_randomized.root'

validation_fraction = 0.3
test_fraction = 0.1

number_of_threads = 8

test = False

if test:
    parameter_values = {
        'hidden_layers_latent' : [3],
        'nodes_latent' : [128],
        'dropout_latent' : [0.5],
        'latent_space' : [256],
        'hidden_layers_output' : [3],
        'nodes_output' : [128],
        'dropout_output' : [0.5],
        'learning_rate' : [1],
        'learning_rate_decay' : [1]
    }

else:
    parameter_values = {
        'hidden_layers_latent' : [3],
        'nodes_latent' : [128],
        'dropout_latent' : [0.5],
        'latent_space' : [256],
        'hidden_layers_output' : [3],
        'nodes_output' : [128],
        'dropout_output' : [0.5],
        'learning_rate' : [1],
        'learning_rate_decay' : [1]
    }

