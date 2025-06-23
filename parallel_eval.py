import os
names = ['dynamic_dance', 'dynamic_fountain', 'dynamic_downstairs', 'dynamic_vehicle', 'dynamic_walk',
         'intense_forward', 'intense_forward2', 'intense_forward3', 'intense_roadside', 'intense_slerp', 
         'mild_building', 'mild_bush', 'mild_rotation', 'mild_rotation2', 'mild_walk']
# names = ['dynamic_fountain', 'dynamic_downstairs', 'dynamic_vehicle']
# names = ['dynamic_dance', 'dynamic_walk',
#          'intense_forward', 'intense_forward2', 'intense_forward3', 'intense_roadside', 'intense_slerp',
#          'mild_building', 'mild_rotation', 'mild_walk']
# names = ['mild_bush', 'mild_rotation2', 'dynamic_fountain', 'dynamic_downstairs', 'dynamic_vehicle']
# names = ['dynamic_dance']
server = 'bayer04'
prefix = 'batch-check'
wandb_prefix = 'batch-check'

tmux_run = False
slurm_run = True
CUDA_offset = 0

cmd_template = "\
      python train.py \
      hydra.run.dir=./exp/{prefix}/{name} \
      +experiment=layered_gavs_overfit \
      dataset.data_path=./gavs-data/dataset/{name} \
    "

for idx, name in enumerate(names):
    cmds = []
    title = ''
    cmds.append(cmd_template.format(prefix=prefix, name=name))
    
    if tmux_run:
        # launch tmux session to run the command in the background with CUDA devices for each task from 0, 1, 2, 3 ...
        # complete_cmd = "tmux new-session -d -s {name} 'CUDA_VISIBLE_DEVICES={idx} {cmd}'".format(name=name, idx=idx+CUDA_offset, cmd=cmd)
        cmd = str.join("\n", cmds)
        complete_cmd = "tmux new-session -d -s {name} 'CUDA_VISIBLE_DEVICES={idx} {cmd}'".format(name=name, idx=idx+CUDA_offset, cmd=cmd)
        print(complete_cmd)
        os.system(complete_cmd)
    elif slurm_run:
        # for cmd in cmds:
        #     complete_cmd = "sbatch -p BayerLab --nodelist=bayer04 -G 1 --cpus-per-task=4 -t \"24:00:00\" --mem=32G --wrap=\"{exec}\" ".format(exec=cmd)
        #     # print(cmd)
        #     os.system(complete_cmd)
        print('there are {} cmds'.format(len(cmds)))
        complete_cmd = "sbatch -p BayerLab --nodelist=\"{server}\" -G 1 --cpus-per-task=4 -t \"4:00:00\" --mem=32G --wrap=\"{exec}\" ".format(server=server, exec=str.join("\n", cmds))
        # print(complete_cmd)
        os.system(complete_cmd)
    else:
        print('CUDA_VISIBLE_DEVICES={idx} {cmd}'.format(idx=CUDA_offset, cmd=str.join('\n', cmds)))
        os.system(str.join('\n', cmds))
