import os
import json
import hydra
import torch
import logging
import sys

# add the project path to the python package path
print('adding path', os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from models.model import GaussianPredictor, to_device
from evaluation.stabilization_evaluator import StabilizationEvaluator
from datasets.util import create_datasets, create_dataloader

def save_image(tensor, path):
    tensor = tensor.squeeze().detach().cpu().clamp(0, 1)
    import torchvision.transforms as transforms
    from PIL import Image
    img_pil = transforms.ToPILImage()(tensor)
    img_pil.save(path)

def get_model_instance(model):
    """
    unwraps model from EMA object
    """
    return model.ema_model if type(model).__name__ == "EMA" else model

def interpolate_camera_K_and_T(K, T_c2w, focal_offset_start, focal_offset_end, z_start, z_end, interpolation_freq):
    fx_tgt = K[0, 0]
    fy_tgt = K[1, 1]
    fx_tgt_start = fx_tgt + fx_tgt * focal_offset_start
    fx_tgt_end = fx_tgt + fx_tgt * focal_offset_end
    fy_tgt_start = fy_tgt + fy_tgt * focal_offset_start
    fy_tgt_end = fy_tgt + fy_tgt * focal_offset_end

    # interpolate focal length
    T_copy = T_c2w.clone()
    fx_tgt = torch.linspace(fx_tgt_start, fx_tgt_end, interpolation_freq)
    fy_tgt = torch.linspace(fy_tgt_start, fy_tgt_end, interpolation_freq)
    z_tgt = torch.linspace(z_start, z_end, interpolation_freq)
    Ks, c2ws = [], []
    for i, (fx, fy) in enumerate(zip(fx_tgt, fy_tgt)):
        R = T_copy[:3, :3]
        forward_v = R[:, 2]
        c2w = T_copy.clone()
        c2w[:3, 3] = c2w[:3, 3] + forward_v * z_tgt[i]
        K_tgt = K.clone()
        K_tgt[0, 0] = fx
        K_tgt[1, 1] = fy
        Ks.append(K_tgt)
        c2ws.append(c2w)
    return Ks, c2ws

def stabilization_evaluate(model, cfg, evaluator, dataloader, save_dir, device=None, depth_min=0.1, depth_max=200, **kwargs):
    save_dir = Path(save_dir)
    model_model = get_model_instance(model)
    model_model.set_eval()

    assert cfg.data_loader.batch_size == 1, "Only batch size 1 is supported for stabilization evaluation" 

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # save rendered images
    logging.info("Generating Images")
    for i, inputs in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            if device is not None:
                to_device(inputs, device)
            target_frame_ids = [0, 1]  # src frame and stabilized frame
            # [data_type, data_name, naming_function, postprocessing]
            # data type: input or output
            # data name: key work in the inputs/outputs dict
            # naming_function: function to generate image names
            # postprocessing: function to apply on the predicted tensor before saving
            target_frame_names = {
                'unstable': [('output', 'unidepth', lambda modality, f_id: (modality), lambda x: (1.0 / (x + 1.0)).clip(0, 1.0).repeat(1, 3, 1, 1)), 
                             ('output', 'color_gauss', lambda modality, f_id: (modality, f_id, 0), lambda x: x), 
                             ('input', 'color', lambda modality, f_id: (modality, f_id, 0), lambda x: x), 
                             ('output', 'depth_gauss', lambda modality, f_id: (modality, f_id, 0), lambda x: (1.0 / (x + 1.0)).clip(0, 1.0).repeat(1, 3, 1, 1))], 
                 'stable': [('output', 'color_gauss', lambda modality, f_id: (modality, f_id, 0), lambda x: x), 
                            ('output', 'depth_gauss', lambda modality, f_id: (modality, f_id, 0), lambda x: (1.0 / (x + 1.0)).clip(0, 1.0).repeat(1, 3, 1, 1)),
                            ('output', 'alpha_gauss', lambda modality, f_id: (modality, f_id, 0), lambda x: x.repeat(1, 3, 1, 1))],
                        }
            
            inputs["target_frame_ids"] = target_frame_ids
            outputs = model(inputs)
            
        for f_id, target_frame_name in zip(target_frame_ids, target_frame_names):
            for src_name, modality, name_fn, fn in target_frame_names[target_frame_name]:
                if src_name == 'output':
                    src = outputs
                elif src_name == 'input':
                    src = inputs

                pred = src[name_fn(modality, f_id)]
                pred = fn(pred)  # apply preprocessing (e.g. depth -> color)
                pred_dir = os.path.join(save_dir, f'{target_frame_name}_{modality}')
                pred_name = inputs[("frame_id", 0)][0]
                if not os.path.exists(pred_dir):
                    os.mkdir(pred_dir)
                save_image(pred, os.path.join(pred_dir, f"{pred_name}.png"))
        
    
    # generate videos from rendered images
    name_template = '%05d.png'

    # export videos
    for target_frame_name in target_frame_names:
        for _, modality, _, _ in target_frame_names[target_frame_name]:
            dir_name = f"{target_frame_name}_{modality}"
            os.system(f"ffmpeg -y -framerate 24 -i {os.path.join(save_dir, dir_name, name_template)} -c:v libx264 -pix_fmt yuv420p {os.path.join(save_dir, dir_name)}.mp4")
            print(f"Video saved to {os.path.join(save_dir, dir_name)}.mp4")

    # quantitative evaluation
    eval_results = evaluator.metrics(save_dir / 'unstable_color', save_dir / 'stable_color_gauss')
    numerical_results = eval_results

    # dump numerical results (only classicial metrics)
    with open(os.path.join(save_dir, 'numerical_results.json'), 'w') as f:
        json.dump(numerical_results, f, indent=4)
    
    print(numerical_results)

@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    print("current directory:", os.getcwd())
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    os.chdir(output_dir)
    print("Working dir:", output_dir)

    # model
    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 1
    model = GaussianPredictor(cfg)
    device = torch.device("cuda:0")
    model.to(device)
    if (ckpt_dir := model.checkpoint_dir()).exists():
        model.load_model(ckpt_dir, ckpt_ids=0)
        print(f"load model from {ckpt_dir}")
    else:
        assert False, 'valid checkpoint not exist.'

    # load predefined depth scale
    if (ckpt_dir.parent / 'depth_scale.json').exists():
        with open(ckpt_dir.parent / 'depth_scale.json', 'r') as f:
            cfg.dataset.predefined_depth_scale = json.load(f)['depth_scale']
            print(f'load predefined depth scale: {cfg.dataset.predefined_depth_scale}')

    # dataset
    split = "test"
    dataset = create_datasets(cfg, split=split)
    dataloader = create_dataloader(cfg, dataset, sampler=None, split=split)

    save_dir = Path(cfg.config.eval_dir)
    evaluator = StabilizationEvaluator()
    stabilization_evaluate(model, cfg, evaluator, dataloader, save_dir, device=device)

if __name__ == "__main__":
    main()