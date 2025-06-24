import torch
import logging
import random
from torch.utils.data import DataLoader, Sampler

from packaging.version import Version
from datasets.gavs_dataset import GaVSDataset

class WindowSampler(Sampler):
    def __init__(self, frame_size, stride_start=-1, stride_delta=2, batch_size=5):
        self.frame_size = frame_size
        self.stride = stride_start
        self.stride_start = stride_start
        self.stride_delta = stride_delta
        self.batch_size = batch_size
        if self.stride < 0:
            self.generate_random_indices()
        else:
            self.generate_indices()

    def __iter__(self):
        return iter(self.flatten_list)

    def __len__(self):
        return len(self.flatten_list)

    def reset(self):
        # reset the sampler to the initial state
        self.stride = self.stride_start
        if self.stride < 0:
            self.generate_random_indices()
        else:
            self.generate_indices()
        print('sampler reset batch size to {}, stride size to {}, sub-group size to {}'.format(self.batch_size, self.stride, len(self.indice_group)))

    def improve_window_size(self):
        # reorganize the indices to reflect the improved window size. 
        self.stride += self.stride_delta
        self.generate_indices()
        print('sampler update batch size to {}, stride size to {}, sub-group size to {}'.format(self.batch_size, self.stride, len(self.indice_group)))

    def generate_random_indices(self):
        # shuffle a list [0, 1, ..., frame_size - 1]
        indices = list(range(self.frame_size))
        random.shuffle(indices)
        # split the indices into sub-groups of size batch_size
        self.indice_group = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        # flatten the list into an 1-d array
        self.flatten_list = []
        for i in self.indice_group:
            self.flatten_list.extend(i)
        # if the last group is not full, remove it
        if len(self.indice_group[-1]) < self.batch_size:
            self.indice_group.pop()
            self.flatten_list = self.flatten_list[:-self.batch_size]

    def generate_indices(self):
        # generate indices for the dataset
        self.indice_group = []
        segment_start = 0
        while True:
            flag = True
            # for p in range(segment_start, segment_start + self.stride):
            indice_subset = [segment_start + i * self.stride for i in range(self.batch_size - 2)]
            
            # if large than the frame size, break
            if indice_subset[-1] >= self.frame_size:
                flag = False
                break
            
            # randomly pick two indices from the whole dataset
            random_one = random.randint(0, self.frame_size - 1)
            random_two = random.randint(0, self.frame_size - 1)
            self.indice_group.append([random_one] + indice_subset + [random_two])
            
            if not flag:
                break
                
            segment_start += (self.batch_size - 2) * self.stride

        # shuffle the indices in the group
        random.shuffle(self.indice_group)
        
        # flatten the list into an 1-d array
        self.flatten_list = []
        for i in self.indice_group:
            self.flatten_list.extend(i)
                

def create_datasets(cfg, split="val"):
    datasets_dict = {
        "gavs": GaVSDataset,
    }[cfg.dataset.name]

    dataset = datasets_dict(cfg, split=split)
    logging.info("There are {:d} {} items\n".format(len(dataset), split)
    )

    if cfg.train.window_sampler and split == "train":
        sampler = sampler = WindowSampler(frame_size=len(dataset), stride_start=1, stride_delta=2, batch_size=cfg.data_loader.batch_size)
    else:
        sampler = None

    if split == "train":
        return dataset, sampler
    else:
        return dataset
    
def create_dataloader(cfg, dataset, sampler, split="val"):
    if cfg.train.window_sampler and split == "train":
        data_loader = DataLoader(
            dataset,
            cfg.data_loader.batch_size,
            num_workers=cfg.data_loader.num_workers,
            pin_memory=True,
            collate_fn=custom_collate,
            sampler=sampler,
            shuffle=False, 
            drop_last=True,
        )   
    else:
        shuffle = True if split == "train" else False
        data_loader = DataLoader(
            dataset,
            cfg.data_loader.batch_size,
            shuffle=shuffle,
            num_workers=cfg.data_loader.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            collate_fn=custom_collate,
        )     
    
    print('data_loader created with batch size {}, num_workers {}, num_batches {}'.format(
        cfg.data_loader.batch_size, 
        cfg.data_loader.num_workers, 
        len(data_loader)
    ))

    return data_loader  



if Version(torch.__version__) < Version("1.11"):
    from torch.utils.data._utils.collate import default_collate
else:
    from torch.utils.data import default_collate


def custom_collate(batch):
    all_keys = batch[0].keys()
    dense_keys = [k for k in all_keys if "sparse" not in k[0]]
    sparse_keys = [k for k in all_keys if "sparse" in k[0]]
    dense_batch = [{k: b[k] for k in dense_keys} for b in batch]
    sparse_batch = {k: [b[k] for b in batch] for k in sparse_keys}
    dense_batch = default_collate(dense_batch)
    batch = sparse_batch | dense_batch
    return batch