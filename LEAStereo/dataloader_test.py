from typing import Dict, Union
import numpy as np
import torch

def main():
    dataset = ToyDataset(10)

    torch.manual_seed(1234)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        num_workers=2
    )
    for batch in loader:
        print(batch["x"].shape, batch["y"])

    # Expected result
    # torch.Size([3, 3, 3]) tensor([2, 1, 3])
    # torch.Size([3, 3, 3]) tensor([6, 7, 9])
    # torch.Size([3, 3, 3]) tensor([5, 4, 8])
    # torch.Size([1, 3, 3]) tensor([0])

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, Union[int, np.ndarray]]:
        return dict(
            x=np.eye(3) * index,
            y=index,
        )

if __name__ == '__main__':
    main()