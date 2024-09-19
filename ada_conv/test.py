from tqdm import tqdm
from dataloader import InfiniteDataLoader

data_list = range(100)
dataloader = InfiniteDataLoader(
    data_list,
    batch_size=8,
    shuffle=True,
    num_workers=4,
)
itr = dataloader.__iter__()
for i in tqdm(range(dataloader.__len__() + 16)):
    data = next(itr)
    assert data.shape[0] == 8, f"{data.shape}"
