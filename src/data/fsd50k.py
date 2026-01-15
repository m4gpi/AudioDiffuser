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
        num_frames = int(self.samples_per_segment / self.sample_rate * metadata.sample_rate)
        waveform, _ = torchaudio.load(str(file_path), num_frames=num_frames)
        return torchaudio.functional.resample(waveform, orig_freq=metadata.sample_rate, new_freq=self.sample_rate).squeeze()

    def _build_metadata(self):
        if self.test is None:
            test_df = pd.read_csv(self.data_dir / self._METADATA_DIR / self._TEST_METADATA)
            test_df["file_path"] = self.data_dir / self._TEST_AUDIO_DIR / test_df["fname"].map(lambda fname: f"{fname}.wav")
            train_df = pd.read_csv(self.data_dir / self._METADATA_DIR / self._TRAIN_VAL_METADATA)
            train_df["file_path"] = self.data_dir /  self._TRAIN_VAL_AUDIO_DIR / train_df["fname"].map(lambda fname: f"{fname}.wav")
            return pd.concat([train_df, test_df], axis=0)
        elif test:
            test_df = pd.read_csv(self.data_dir / self._METADATA_DIR / self._TEST_METADATA)
            test_df["file_path"] = self.data_dir / self._TEST_AUDIO_DIR / test_df["fname"].map(lambda fname: f"{fname}.wav")
            return test_df
        else:
            train_df = pd.read_csv(self.data_dir / self._METADATA_DIR / self._TRAIN_VAL_METADATA)
            train_df["file_path"] = self.data_dir /  self._TRAIN_VAL_AUDIO_DIR / train_df["fname"].map(lambda fname: f"{fname}.wav")
            return train_df

class FSD50KDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/fsd50k",
        sample_rate: int = 44100,
        segment_len: int = 3.072,
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
        self.data: Dataset | None = None
        self.train: Dataset | None = None
        self.val: Dataset | None = None
        self.test: Dataset | None = None

    def prepare_data(self):
        FSD50K(data_dir=self.data_dir, download=True)

    def setup(self, stage: Optional[str] = None):
        self.data = FSD50K(
            data_dir=self.data_dir,
            sample_rate=self.sample_rate,
            segment_len=self.segment_len,
        )
        self.train, self.val, self.test = torch.utils.data.random_split(
            self.data,
            (1 - self.val_prop - self.test_prop, self.val_prop, self.test_prop),
            generator=self.generator
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
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
