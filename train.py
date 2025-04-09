import torch
import matplotlib.pyplot as plt
from loader import FernDataset, NormalizeFeatures, data_split
from models import FernModel, TraitsPredictor
from tqdm import trange
from pathlib import Path
import pytorch_lightning as pl

torch.set_float32_matmul_precision('medium')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
pl.seed_everything(seed)
torch.cuda.manual_seed(seed)

norm_transform = NormalizeFeatures()
dataset = FernDataset(Path('data/Ferns'), transform=norm_transform)
data = dataset[0]

model = TraitsPredictor(in_traits=data.x_species.size(1), in_phylo=data.x_species_phylo.size(1), 
                        in_space=data.global_data.size(1), hidden_channels=16, out_channels=dataset.y_index.size, 
                        num_layers=2)

train_data, test_data = data_split(data)
    
model = model.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()
Loss = []
test_losses = []
for epoch in trange(200):
    model.train()
    optimizer.zero_grad()
    out = model(train_data)
    out = out*~train_data.x_species_traits_nanmask
    loss = loss_fn(out, train_data.y)
    loss.backward()
    optimizer.step()
    Loss.append(loss.item())
    
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            out = model(test_data)
            out = out*~test_data.x_species_traits_nanmask
            test_loss = loss_fn(out, test_data.y)
            test_losses.append(test_loss.item())

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax in axs:
        ax.plot(Loss, label='Train Loss')
        ax.plot(range(0, epoch+1, 5), test_losses, label='Test Loss')
        ax.legend()

    ax.set_yscale('log')
    fig.savefig('tmp_train.png')
    plt.close(fig)

# Save the model
torch.save(model.state_dict(), 'graph_model.pth')