import os
import time
import logging
import torch
import hydra
import torch.optim as optim
import json
import wandb
from rich.progress import Progress

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning import seed_everything
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy

from datasets.util import create_datasets, create_dataloader
from trainer import Trainer

def run_epoch(fabric,
              trainer,
              train_loader,
              optimiser,
              lr_scheduler,
              sampler):
    """Run a single epoch of training and validation
    """
    cfg = trainer.cfg
    trainer.model.set_train()

    if fabric.is_global_zero:
        logging.info("Training on epoch {}".format(trainer.epoch))

    progress = Progress()
    progress.start()
    task = progress.add_task(f"[green]Training Epoch {trainer.epoch}", total=len(train_loader))

    for batch_idx, inputs in enumerate(train_loader):
        inputs["target_frame_ids"] = cfg.model.gauss_novel_frames
        losses, outputs = trainer(inputs, sampler)

        optimiser.zero_grad(set_to_none=True)
        fabric.backward(losses["loss/total"])
        optimiser.step()        
        step = trainer.step

        progress.update(task, advance=1, description=f"Loss: {losses['loss/total']:.4f}")

        log_visual = trainer.step % trainer.cfg.run.log_frequency == 0
        if fabric.is_global_zero:
            learning_rate = lr_scheduler.get_last_lr()
            if type(learning_rate) is list:
                learning_rate = max(learning_rate)
            
            # save the loss and scales
            trainer.log_scalars("train", outputs, losses, learning_rate)

            # save the visual results
            if log_visual:
                trainer.log("train", inputs, outputs)
                
            # save the model
            if step % cfg.run.save_frequency == 0 and step != 0:
                trainer.model.save_model(optimiser, step)
            lr_scheduler.step()

        if log_visual:
            torch.cuda.empty_cache()
            
        trainer.step += 1
    
    progress.stop()

@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    
    # set up the output directory
    output_dir = hydra_cfg['runtime']['output_dir']
    logging.info(f"Working dir: {output_dir}")
    
    # get absolute path for acquiring pretrained weights and dataset loader
    original_cwd = hydra.utils.get_original_cwd()
    cfg.train.load_weights_folder = os.path.join(original_cwd, cfg.train.load_weights_folder)
    cfg.dataset.data_path = os.path.join(original_cwd, cfg.dataset.data_path)
    
    # set up random set
    torch.set_float32_matmul_precision('high')
    seed_everything(cfg.run.random_seed)
    
    # set up training precision
    fabric = Fabric(
        accelerator="cuda",
        devices=cfg.train.num_gpus,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=cfg.train.mixed_precision
    )
    fabric.launch()
    fabric.barrier()

    # set up model
    trainer = Trainer(cfg)
    model = trainer.model
   
    # set up optimiser
    optimiser = optim.Adam(model.parameters_to_train, cfg.optimiser.learning_rate)
    
    # set up checkpointing
    if (ckpt_dir := model.checkpoint_dir()).exists():
        # resume training
        model.load_model(ckpt_dir, optimiser=optimiser)
    elif cfg.train.load_weights_folder:
        model.load_model(cfg.train.load_weights_folder)

    # set up datasets and dataloaders
    trainer, optimiser = fabric.setup(trainer, optimiser)
    train_dataset, sampler = create_datasets(cfg, split="train")
    train_loader = create_dataloader(cfg, train_dataset, sampler, split="train")

    # count varied step number in all epochs
    def count_step_number(dataset, sampler):
        cnt = 0
        for e in range(cfg.optimiser.num_epochs):
            loader = create_dataloader(cfg, dataset, sampler, split="train")
            cnt += len(loader)
            sampler.improve_window_size()
        print(f"Total number of steps: {cnt}")
        sampler.reset()
        return cnt

    # export depth scale json
    if hasattr(train_dataset, "get_depth_scale"):
        with open("depth_scale.json", "w") as f:
            print(f'Exporting depth scale to {f.name}')
            json.dump(train_dataset.get_depth_scale(), f)

    train_loader = fabric.setup_dataloaders(train_loader)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=count_step_number(train_dataset, sampler))

    # init wandb 
    if fabric.is_global_zero:
        if cfg.train.logging:
            display_name = output_dir.split('/')[-2] + '-' + output_dir.split('/')[-1]
            wandb.init(project='GaVS', name=display_name)
            trainer.set_logger(wandb)
    
    # launch training
    trainer.epoch = 0
    trainer.start_time = time.time()
    print('staring training, num_epochs:', cfg.optimiser.num_epochs)
    for trainer.epoch in range(cfg.optimiser.num_epochs):
        run_epoch(fabric, trainer, train_loader, optimiser, lr_scheduler, sampler)
        if sampler:
            sampler.improve_window_size()
            train_loader = create_dataloader(cfg, train_dataset, sampler, split="train") # reinitialize dataloader with new sampler

    # save last checkpoint
    trainer.model.save_model(optimiser, trainer.step)

    # run evaluation
    run_evaluation(cfg, original_cwd)

def run_evaluation(cfg, original_cwd):
    # get current absoluate path of working directory
    current_dir = os.getcwd()
    # switch to project directory
    os.chdir(original_cwd)

    template = "python evaluate.py \
        train.mode='eval' \
        hydra.run.dir={OUTPUT_DIR} \
        +experiment=layered_gavs_eval \
        dataset.data_path={DATA_DIR} \
    "

    print("Running evaluation")
    eval_cmd = template.format(
        OUTPUT_DIR=current_dir,
        DATA_DIR=cfg.dataset.data_path,
    )
    print(eval_cmd)
    os.system(eval_cmd)

    # wandb log evaluation results
    result_path = current_dir + '/visualization'
    # open the json file named 'numerical_results.json' and log the results
    with open(result_path + '/numerical_results.json', 'r') as f:
        results = json.load(f)
        results = {'eval_scalar/' + k: v for k, v in results.items() if 'eval' in k}
        wandb.log(results)
    
    # list all mp4 files in the visualization directory and log them to wandb with prefix 'eval_video/'
    for file in os.listdir(result_path):
        if file.endswith('.mp4'):
            wandb.log({'eval_video/' + file: wandb.Video(result_path + '/' + file)})

if __name__ == "__main__":
    main()
