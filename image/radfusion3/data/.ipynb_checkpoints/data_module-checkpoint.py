import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .. import builder


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, test_split="test"):
        super().__init__()
        self.cfg = cfg
        self.dataset = builder.build_dataset(cfg)
        self.test_split = test_split

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)

        if self.cfg.dataset.weighted_sample:
            sampler = dataset.get_sampler()
        elif self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(dataset, shuffle=True)
        else:
            sampler = None

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=(sampler is None),
            sampler=sampler,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.trainer.num_workers,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "val")
        dataset = self.dataset(self.cfg, split="val", transform=transform)

        sampler = DistributedSampler(dataset, shuffle=False) if self.trainer and self.trainer.world_size > 1 else None

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False if sampler else True,
            sampler=sampler,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.trainer.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, self.test_split)
        dataset = self.dataset(self.cfg, split=self.test_split, transform=transform)

        sampler = DistributedSampler(dataset, shuffle=False) if self.trainer and self.trainer.world_size > 1 else None

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False if sampler else True,
            sampler=sampler,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.trainer.num_workers,
        )

    def all_dataloader(self):
        transform = builder.build_transformation(self.cfg, "all")
        dataset = self.dataset(self.cfg, split=self.cfg.test_split, transform=transform)

        sampler = DistributedSampler(dataset, shuffle=False) if self.trainer and self.trainer.world_size > 1 else None

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False if sampler else True,
            sampler=sampler,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.trainer.num_workers,
        )
