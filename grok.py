import torch
from torch import tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import numpy as np

import dataset
import training

epochs = 1000
val_split = .2
batch_size = 2
random_seed = 42
d_model=2
num_heads=2

class Transformer(torch.nn.Module):
    def __init__(self, d_model, numheads, out_dim):
        super().__init__()
        self.trans = torch.nn.Transformer(num_encoder_layers=0, num_decoder_layers=4, d_model=d_model, dim_feedforward=d_model, nhead=numheads)
        self.lin = torch.nn.Linear(d_model, out_dim, bias=False)

    def forward(self, x):
        y = self.trans(x, x)
        return self.lin(y)

ds_x, ds_y, vocab_size = dataset.create_dataset_v2(dataset.addition_mod_p, np.arange(97), np.arange(97), dataset.make_dict_v2(97), 97) 
ds = TensorDataset(tensor(ds_x, dtype=torch.float32), tensor(torch.nn.functional.one_hot(tensor(ds_y), num_classes=vocab_size), dtype=torch.float32))
dataset_size = len(ds)
indices = list(range(dataset_size))
split = int(np.floor(val_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
training_loader = DataLoader(ds, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(ds, batch_size=batch_size, sampler=valid_sampler)

# Default log dir is './runs'
writer = SummaryWriter()

model = Transformer(d_model, num_heads, vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # TODO : adjust
loss_fn = torch.nn.CrossEntropyLoss()

training.train(model, training_loader, validation_loader, optimizer, loss_fn, epochs, writer)
print("Done training!")