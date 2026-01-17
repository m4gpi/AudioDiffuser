from typing import Any, Dict, Optional

import numpy as np
import pathlib
import torch
import torchaudio
import librosa
import pandas as pd

from pytorch_lightning import LightningDataModule

class FSD50K(torch.utils.data.Dataset):
    _TRAIN_VAL_AUDIO_DIR = "FSD50K.dev_audio"
    _TEST_AUDIO_DIR = "FSD50K.eval_audio"
    _METADATA_DIR: str = "FSD50K.ground_truth"
    _TRAIN_VAL_METADATA: str = "dev.csv"
    _TEST_METADATA: str = "eval.csv"

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 44_100,
        segment_len: int = 1.536,
        download: bool = False,
        test: bool | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        self.test = test
        self.sample_rate = sample_rate
        self.segment_len = segment_len
        self.metadata = self._build_metadata()
        self.metadata["duration_seconds"] = self.metadata.file_path.map(lambda path: librosa.get_duration(path=path))
        self.metadata = self.metadata[self.metadata["duration_seconds"] > self.segment_len]
        self.x = self.metadata.file_path.to_numpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        file_path = self.x[idx]
        return self.load_sample(file_path)

    @property
    def samples_per_segment(self):
        return int(self.sample_rate * self.segment_len)

    def load_sample(self, file_path: pathlib.Path):
        metadata = torchaudio.info(str(file_path))
        num_frames = int(metadata.sample_rate * self.segment_len)
        max_offset = metadata.num_frames - num_frames
        frame_offset = np.random.choice(np.arange(max_offset))
        waveform, _ = torchaudio.load(str(file_path), frame_offset=frame_offset, num_frames=num_frames)
        return torchaudio.functional.resample(waveform, orig_freq=metadata.sample_rate, new_freq=self.sample_rate).squeeze()

    def _build_metadata(self):
        if self.test == True:
            test_df = pd.read_csv(self.data_dir / self._METADATA_DIR / self._TEST_METADATA)
            test_df["file_path"] = self.data_dir / self._TEST_AUDIO_DIR / test_df["fname"].map(lambda fname: f"{fname}.wav")
            return test_df
        elif self.test == False:
            train_df = pd.read_csv(self.data_dir / self._METADATA_DIR / self._TRAIN_VAL_METADATA)
            train_df["file_path"] = self.data_dir /  self._TRAIN_VAL_AUDIO_DIR / train_df["fname"].map(lambda fname: f"{fname}.wav")
            return train_df
        else:
            test_df = pd.read_csv(self.data_dir / self._METADATA_DIR / self._TEST_METADATA)
            test_df["file_path"] = self.data_dir / self._TEST_AUDIO_DIR / test_df["fname"].map(lambda fname: f"{fname}.wav")
            train_df = pd.read_csv(self.data_dir / self._METADATA_DIR / self._TRAIN_VAL_METADATA)
            train_df["file_path"] = self.data_dir /  self._TRAIN_VAL_AUDIO_DIR / train_df["fname"].map(lambda fname: f"{fname}.wav")
            return pd.concat([train_df, test_df], axis=0)

class FrameSampler(torch.utils.data.Sampler):
    def __init__(self, data: torch.utils.data.Dataset) -> None:
        duration_seconds = data.metadata["duration_seconds"].to_numpy()
        self.num_samples = len(data)
        self.weights = torch.tensor(duration_seconds / duration_seconds.sum())

    def __iter__(self):
        sampled_indices = torch.multinomial(self.weights, self.num_samples, replacement=True)
        return iter(sampled_indices.tolist())

class FSD50KDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/fsd50k",
        sample_rate: int = 32000,
        segment_len: int = 1.536,
        batch_size: int = 64,
        val_prop: float = 0.2,
        test_prop: float = 0.2,
        num_workers: int = 4,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.batch_size = batch_size
        self.stft_args = dict(n_fft=n_fft, hop_length=hop_length, center=True)
        self.data: torch.utils.data.Dataset | None = None
        self.train: torch.utils.data.Dataset | None = None
        self.val: torch.utils.data.Dataset | None = None
        self.test: torch.utils.data.Dataset | None = None

    def prepare_data(self):
        pass
        # FSD50K(data_dir=self.hparams.data_dir, download=True)

    def setup(self, stage: Optional[str] = None):
        self.data = FSD50K(
            data_dir=self.hparams.data_dir,
            sample_rate=self.hparams.sample_rate,
            segment_len=self.hparams.segment_len,
        )
        # self.train, self.val, self.test = torch.utils.data.random_split(
        #     self.data,
        #     (1 - self.hparams.val_prop - self.hparams.test_prop, self.hparams.val_prop, self.hparams.test_prop),
        #     generator=self.generator
        # )

    def train_dataloader(self):
        dl = torch.utils.data.DataLoader(
            dataset=self.data,
            batch_size=6,
            sampler=FrameSampler(data=self.data),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
        while True:
            yield from dl

    def val_dataloader(self):
        #NOTE: Always eeturn ANY dataloader. Lightning just needs to see it exists
        dummy_dataset = torch.utils.data.TensorDataset(torch.zeros(1))
        val_dl = torch.utils.data.DataLoader(dummy_dataset, batch_size=1)
        return val_dl

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "sc09.yaml")
    cfg.data_dir = str(root / "sc09")
    _ = hydra.utils.instantiate(cfg)
