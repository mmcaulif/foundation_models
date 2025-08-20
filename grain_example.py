import grain.python as pygrain
import numpy as np

SEED = 42

class ExampleDataset:
    def __init__(self):
        self.x = np.ones((1000, 10))
        self.y = np.ones((1000,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
    
data_source = ExampleDataset()

sampler = pygrain.IndexSampler(
    len(data_source),
    shuffle=True,
    seed=SEED,
    shard_options=pygrain.NoSharding(),
    num_epochs=1,
)

batch_size = 256
dl = pygrain.DataLoader(
    data_source=data_source, sampler=sampler, operations=[pygrain.Batch(batch_size)]
)

for i, (x, y) in enumerate(dl):
    print(i, x.shape, y.shape)