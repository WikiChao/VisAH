"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch as th
import torch.nn as nn
import math
from scipy.signal import hilbert
import torchaudio as ta

class FourierTransform:
    def __init__(self,
                 fft_bins=2048,
                 win_length_ms=40,
                 frame_rate_hz=100,
                 causal=False,
                 preemphasis=0.0,
                 sample_rate=48000,
                 normalized=False):
        self.sample_rate = sample_rate
        self.frame_rate_hz = frame_rate_hz
        self.preemphasis = preemphasis
        self.fft_bins = fft_bins
        self.win_length = int(sample_rate * win_length_ms / 1000)
        self.hop_length = int(sample_rate / frame_rate_hz)
        self.causal = causal
        self.normalized = normalized
        if self.win_length > self.fft_bins:
            print('FourierTransform Warning: fft_bins should be larger than win_length')

    def _convert_format(self, data, expected_dims):
        if not type(data) == th.Tensor:
            data = th.Tensor(data)
        if len(data.shape) < expected_dims:
            data = data.unsqueeze(0)
        if not len(data.shape) == expected_dims:
            raise Exception(f"FourierTransform: data needs to be a Tensor with {expected_dims} dimensions but got shape {data.shape}")
        return data

    def _preemphasis(self, audio):
        if self.preemphasis > 0:
            return th.cat((audio[:, 0:1], audio[:, 1:] - self.preemphasis * audio[:, :-1]), dim=1)
        return audio

    def _revert_preemphasis(self, audio):
        if self.preemphasis > 0:
            for i in range(1, audio.shape[1]):
                audio[:, i] = audio[:, i] + self.preemphasis * audio[:, i-1]
        return audio

    def _magphase(self, complex_stft):
        mag, phase = ta.functional.magphase(complex_stft, 1.0)
        return mag, phase

    def stft(self, audio):
        '''
        wrapper around th.stft
        audio: wave signal as th.Tensor
        '''
        hann = th.hann_window(self.win_length)
        hann = hann.cuda() if audio.is_cuda else hann
        # spec = th.stft(audio, n_fft=self.fft_bins, hop_length=self.hop_length, win_length=self.win_length,
        #                window=hann, center=not self.causal, normalized=self.normalized, return_complex=False)
        spec = th.stft(audio, n_fft=self.fft_bins, hop_length=self.hop_length, win_length=self.win_length,
                       window=hann, center=not self.causal, normalized=self.normalized, return_complex=True)
        spec = th.view_as_real(spec)
        return spec.contiguous()

    def complex_spectrogram(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: th.Tensor of size channels x frequencies x time_steps (channels x y_axis x x_axis)
        '''
        self._convert_format(audio, expected_dims=2)
        audio = self._preemphasis(audio)
        return self.stft(audio)

    def magnitude_phase(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: tuple containing two th.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        stft = self.complex_spectrogram(audio)
        return self._magphase(stft)

    def mag_spectrogram(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: magnitude spectrum as th.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        return self.magnitude_phase(audio)[0]

    def power_spectrogram(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: power spectrum as th.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        return th.pow(self.mag_spectrogram(audio), 2.0)

    def phase_spectrogram(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: phase spectrum as th.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        return self.magnitude_phase(audio)[1]

    def mel_spectrogram(self, audio, n_mels):
        '''
        audio: wave signal as th.Tensor
        n_mels: number of bins used for mel scale warping
        return: mel spectrogram as th.Tensor of size channels x n_mels x time_steps for magnitude and phase spectrum
        '''
        spec = self.power_spectrogram(audio)
        mel_warping = ta.transforms.MelScale(n_mels, self.sample_rate)
        return mel_warping(spec)

    def complex_spec2wav(self, complex_spec, length):
        '''
        inverse stft
        complex_spec: complex spectrum as th.Tensor of size channels x frequencies x time_steps x 2 (real part/imaginary part)
        length: length of the audio to be reconstructed (in frames)
        '''
        complex_spec = self._convert_format(complex_spec, expected_dims=4)
        hann = th.hann_window(self.win_length)
        hann = hann.cuda() if complex_spec.is_cuda else hann
        wav = ta.functional.istft(complex_spec, n_fft=self.fft_bins, hop_length=self.hop_length, win_length=self.win_length, window=hann, length=length, center=not self.causal)
        wav = self._revert_preemphasis(wav)
        return wav

    def magphase2wav(self, mag_spec, phase_spec, length):
        '''
        reconstruction of wav signal from magnitude and phase spectrum
        mag_spec: magnitude spectrum as th.Tensor of size channels x frequencies x time_steps
        phase_spec: phase spectrum as th.Tensor of size channels x frequencies x time_steps
        length: length of the audio to be reconstructed (in frames)
        '''
        mag_spec = self._convert_format(mag_spec, expected_dims=3)
        phase_spec = self._convert_format(phase_spec, expected_dims=3)
        complex_spec = th.stack([mag_spec * th.cos(phase_spec), mag_spec * th.sin(phase_spec)], dim=-1)
        return self.complex_spec2wav(complex_spec, length)

class Loss(nn.Module):
    def __init__(self, mask_beginning=0, mask_end=0):
        '''
        base class for losses that operate on the wave signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__()
        self.mask_beginning = mask_beginning
        self.mask_end = mask_end

    def forward(self, data, target, **kwargs):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data_len = data.shape[-1]
        data = data[..., self.mask_beginning : data_len - self.mask_end]
        target = target[..., self.mask_beginning : data_len - self.mask_end]
        return self._loss(data, target, **kwargs)

    def _loss(self, data, target):
        raise NotImplementedError

class Envelop_distance(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        device = data.device
        num_channels = data.shape[1]
        dist = []
        for i in range(num_channels):
            data_ = np.abs(hilbert(data[:, i].detach().cpu().numpy()))
            target_ = np.abs(hilbert(target[:, i].detach().cpu().numpy()))
            dist_ = np.sqrt(np.mean((data_ - target_)**2))
            dist.append(dist_)
        return th.Tensor(dist).sum().to(device)

class L1Loss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        return th.mean(th.abs(data - target))


class L2Loss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        return th.mean((data - target).pow(2))


class NormWeightedL2Loss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        assert len(target.shape) == 3
        weight = th.abs(target)
        return th.mean((data - target).pow(2) * weight) * min(5., 1/weight.max())


class WeightedL1Loss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        assert len(target.shape) == 3
        weight = th.abs(target)
        return th.mean(th.abs(data - target) * weight)


class WeightedL2Loss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        assert len(target.shape) == 3
        weight = th.abs(target)
        return th.mean((data - target).pow(2) * weight)


class AmplitudeLoss(Loss):
    def __init__(self, sample_rate, mask_beginning=0, mask_end=0, log=False):
        '''
        :param sample_rate: (int) sample rate of the audio signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__(mask_beginning, mask_end)
        self.log = log
        self.eps = 1e-8
        self.fft = FourierTransform(sample_rate=sample_rate)

    def _transform(self, data):
        return self.fft.stft(data.view(-1, data.shape[-1]))

    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data, target = self._transform(data), self._transform(target)
        data = th.sqrt(th.sum(data**2, dim=-1) + self.eps)
        target = th.sqrt(th.sum(target**2, dim=-1) + self.eps)
        if self.log:
            data = th.log(data + 1.0)
            target = th.log(target + 1.0)
        return th.mean(th.abs(data - target))


class PhaseLoss(Loss):
    def __init__(
        self,
        sample_rate,
        mask_beginning=0,
        mask_end=0,
        fft_bins=2048,
        ignore_below=0.1,
        silent_threshold=0.001,
        weighted=False
    ):
        '''
        :param sample_rate: (int) sample rate of the audio signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__(mask_beginning, mask_end)
        self.ignore_below = ignore_below
        self.silent_threshold = silent_threshold
        self.weighted = weighted
        self.eps = 1e-8
        fft_bins_ref = 2048
        win_length_ms_ref = 40
        frame_rate_hz_ref = 100
        fft_params = {'fft_bins': fft_bins}
        fft_params['win_length_ms'] = win_length_ms_ref * fft_bins / fft_bins_ref
        fft_params['frame_rate_hz'] = frame_rate_hz_ref * fft_bins_ref / fft_bins
        fft_params['sample_rate'] = sample_rate
        self.fft = FourierTransform(**fft_params)

    @staticmethod
    def _get_silence_mask(audio, threshold, window_size=65):
        pooling = nn.MaxPool1d(window_size, stride=1)
        pad_len = window_size//2
        mask = pooling(nn.functional.pad(audio.abs(), (pad_len, pad_len), mode='replicate')) < threshold

        return mask

    def _transform(self, data):
        return self.fft.stft(data.view(-1, data.shape[-1]))

    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data = data.clone()
        target = target.clone()
        silence_mask = self._get_silence_mask(target, self.silent_threshold)
        data[silence_mask] = 0
        target[silence_mask] = 0

        data, target = self._transform(data).view(-1, 2), self._transform(target).view(-1, 2)
        # ignore low energy components for numerical stability
        target_energy = th.sqrt(th.sum(target**2, dim=-1) + self.eps)
        pred_energy = th.sqrt(th.sum(data.detach()**2, dim=-1) + self.eps)
        target_mask = target_energy > self.ignore_below * th.mean(target_energy)
        pred_mask = pred_energy > self.ignore_below * th.mean(target_energy)
        indices = th.nonzero(target_mask * pred_mask).view(-1)
        if len(indices) == 0:
            return th.ones(1)[0].to(data.device) * th.pi
        data, target = th.index_select(data, 0, indices), th.index_select(target, 0, indices)
        target_energy = th.index_select(target_energy, 0, indices)
        # compute actual phase loss in angular space
        data_angles, target_angles = th.atan2(data[:, 0], data[:, 1]), th.atan2(target[:, 0], target[:, 1])
        loss = th.abs(data_angles - target_angles)
        # positive + negative values in left part of coordinate system cause angles > pi
        # => 2pi -> 0, 3/4pi -> 1/2pi, ... (triangle function over [0, 2pi] with peak at pi)
        loss = np.pi - th.abs(loss - np.pi)
        if self.weighted:
            return th.mean(loss) * th.mean(target_energy)
        else:
            return th.mean(loss)


class MultiResolutionSTFTLoss(Loss):
    """Multi resolution STFT loss module."""
    def __init__(
        self,
        sample_rate,
        mask_beginning=0,
        mask_end=0,
        log=False,
        fft_bins_list=[256, 128, 64, 32],
        weighted=False
    ):
        super().__init__(mask_beginning, mask_end=mask_end)
        self.log = log
        self.eps = 1e-8
        self.weighted = weighted

        assert len(fft_bins_list) >= 3
        fft_bins_ref = 2048
        win_length_ms_ref = 40
        frame_rate_hz_ref = 100
        self.fft_list = []
        for fft_bins in fft_bins_list:
            curr_fft_params = {'fft_bins': fft_bins}
            curr_fft_params['win_length_ms'] = win_length_ms_ref * fft_bins / fft_bins_ref
            curr_fft_params['frame_rate_hz'] = frame_rate_hz_ref * fft_bins_ref / fft_bins
            curr_fft_params['sample_rate'] = sample_rate
            self.fft_list.append(FourierTransform(**curr_fft_params))

    def _loss(self, data, target):
        data = data.view(-1, data.shape[-1])  # (B, C, T) -> (B x C, T)
        target = target.view(-1, target.shape[-1])  # (B, C, T) -> (B x C, T)
        mag_loss = 0.0
        for fft in self.fft_list:
            data_stft = fft.stft(data)
            target_stft = fft.stft(target)
            data_stft = th.sqrt(th.sum(data_stft**2, dim=-1) + self.eps)
            target_stft = th.sqrt(th.sum(target_stft**2, dim=-1) + self.eps)
            if self.log:
                data_stft = th.log(data_stft + 1.0)
                target_stft = th.log(target_stft + 1.0)
            if self.weighted:
                mag_loss += th.mean(th.abs(data_stft - target_stft) * target_stft)
            else:
                mag_loss += th.mean(th.abs(data_stft - target_stft))
        
        mag_loss /= len(self.fft_list)

        return mag_loss


class STFTLoss(Loss):
    def __init__(
        self,
        sample_rate,
        mask_beginning=0,
        mask_end=0,
        log=False,
        fft_bins=2048,
        weighted=False
    ):
        super().__init__(mask_beginning, mask_end=mask_end)
        self.log = log
        self.eps = 1e-8
        self.weighted = weighted

        fft_bins_ref = 2048
        win_length_ms_ref = 40
        frame_rate_hz_ref = 100
        fft_params = {'fft_bins': fft_bins}
        fft_params['win_length_ms'] = win_length_ms_ref * fft_bins / fft_bins_ref
        fft_params['frame_rate_hz'] = frame_rate_hz_ref * fft_bins_ref / fft_bins
        fft_params['sample_rate'] = sample_rate
        self.fft = FourierTransform(**fft_params)

    def _loss(self, data, target):
        data = data.view(-1, data.shape[-1])  # (B, C, T) -> (B x C, T)
        target = target.view(-1, target.shape[-1])  # (B, C, T) -> (B x C, T)
        data_stft = self.fft.stft(data)
        target_stft = self.fft.stft(target)
        data_stft = th.sqrt(th.sum(data_stft**2, dim=-1) + self.eps)
        target_stft = th.sqrt(th.sum(target_stft**2, dim=-1) + self.eps)
        if self.log:
            data_stft = th.log(data_stft + 1.0)
            target_stft = th.log(target_stft + 1.0)
        if self.weighted:
            mag_loss = th.mean(th.abs(data_stft - target_stft) * target_stft)
        else:
            mag_loss = th.mean(th.abs(data_stft - target_stft))

        return mag_loss

class SDRloss(Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _loss(self, data, ref):
        return -th.mean(th.log10(th.mean(ref**2, dim=-1) / th.mean((ref-data)**2, dim=-1))) * 10
