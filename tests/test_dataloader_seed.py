from types import SimpleNamespace

from torch.utils.data import Dataset

from ultralytics.data.build import build_dataloader


class RangeDataset(Dataset):
    def __init__(self, n: int = 16, seed: int = 0):
        self.n = n
        self.hyp = SimpleNamespace(seed=seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        return index


def collect_epoch_order(seed: int) -> list[int]:
    loader = build_dataloader(RangeDataset(seed=seed), batch=4, workers=0, shuffle=True)
    order = []
    for batch in loader:
        order.extend(batch.tolist())
    return order


def test_build_dataloader_respects_user_seed():
    assert collect_epoch_order(7) == collect_epoch_order(7)
    assert collect_epoch_order(7) != collect_epoch_order(8)
