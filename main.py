import torch
from gformer import Gformer

model = Gformer(
        n_outputs=54,
        n_chans=62,
        sfreq=250,
        n_times=500,
        heads=2,
        depth=1,
        emb_size=128,
        kernel_size=64,
        depth_multiplier=3,
        pool_size = 10,
        drop_prob_cnn=0.5,
        drop_prob_posi=0.5,
        drop_prob_final=0.5,
    ).eval()

data = torch.randn(16, 62, 500) # (B, C, T)
preds = model(data)
print(preds.shape)