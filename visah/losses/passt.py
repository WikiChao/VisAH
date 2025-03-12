from hear21passt.base import get_basic_model,get_model_passt
import torch
from torch import nn
import torchaudio


class PaSST(nn.Module):
    def __init__(self, device):
        super(PaSST, self).__init__()
        self.model = get_basic_model(mode="logits")
        self.model.to(device)
        self.device = device

    def forward(self, file_paths, device):
        # load audio files at 32k sampling rate
        audio_wave = torch.stack([torchaudio.load(file_path)[0] for file_path in file_paths]).squeeze(1)
        # resample to 32k
        audio_wave = torchaudio.transforms.Resample(orig_freq=44100, new_freq=32000)(audio_wave)
        audio_wave = audio_wave.to(device)
        with torch.no_grad():
            logits = self.model(audio_wave)
        return logits