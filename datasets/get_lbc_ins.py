import os
root = '/data/carla_dataset/data_collection/interactive'
count, total_count = 0, 0
for basic_scenario in os.listdir(root):
    for variant_scenario in os.listdir(os.path.join(root, basic_scenario, 'variant_scenario')):
        
        if os.path.isdir(os.path.join(root, basic_scenario, 'variant_scenario', variant_scenario, 'instance_segmentation/lbc_ins')):
            print(os.path.join(root, basic_scenario))
            count += 1
        total_count += 1
print(count)
print(total_count)

