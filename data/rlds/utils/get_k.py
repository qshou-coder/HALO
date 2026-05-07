def get_goal_index(dataset_name):
    mixture ={
        "fractal20220817_data": 10,                # Google RT-1 Robot Data (Large-Scale)
        "kuka": 30, 
        "bridge": 10,                              # Original Version of Bridge V2 from Project Website
        "jaco_play": 30,    
        "berkeley_cable_routing": 20, 
        "roboturk": 4, 
        "viola": 30, 
        "berkeley_autolab_ur5": 10, 
        "toto": 60, 
        "language_table": 6, 
        "stanford_hydra_dataset_converted_externally_to_rlds": 24, 
        "austin_buds_dataset_converted_externally_to_rlds": 40, 
        "nyu_franka_play_dataset_converted_externally_to_rlds": 6, 
        "furniture_bench_dataset_converted_externally_to_rlds": 20, 
        "ucsd_kitchen_dataset_converted_externally_to_rlds": 4, 
        "austin_sailor_dataset_converted_externally_to_rlds": 40, 
        "austin_sirius_dataset_converted_externally_to_rlds": 40, 
        "dlr_edan_shared_control_converted_externally_to_rlds": 10, 
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds": 60, 
        "utaustin_mutex": 44, 
        "berkeley_fanuc_manipulation": 30, 
        # "cmu_stretch": 100, 
    }


    assert dataset_name in mixture
    return mixture[dataset_name]