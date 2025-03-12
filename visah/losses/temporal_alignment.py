from scipy.stats import wasserstein_distance
import numpy as np
import torch

class WassersteinDistance:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, y_true, y_pred):
        """
        Args: y_true: ground truth audio tensor, shape[batch_size, n_channels, n_frames]
                y_pred: predicted audio tensor, shape[batch_size, n_channels, n_frames]
        """
        # transform tensor to numpy array
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        n_frames = y_true.shape[2]
        y_true = y_true.reshape(-1, n_frames)
        y_pred = y_pred.reshape(-1, n_frames)
        n_channels = y_true.shape[0]
        wasserstein_dist = 0
        for i in range(n_channels):
            wasserstein_dist += wasserstein_distance(y_true[i], y_pred[i])
        # transform back to tensor
        wasserstein_dist = torch.tensor(wasserstein_dist)
        return wasserstein_dist / n_channels

if __name__ == '__main__':
    output_dir = 'test_dir'
    import torchaudio
    # read audio files
    y_true, sr = torchaudio.load(f'{output_dir}/target_audio.wav')
    y_pred, sr = torchaudio.load(f'{output_dir}/pred_audio.wav')
    y_input, sr = torchaudio.load(f'{output_dir}/input_audio.wav')
    wasserstein_dist = WassersteinDistance()
    dtw_dist = DTWDistance()
    print(wasserstein_dist(y_true.unsqueeze(0), y_pred.unsqueeze(0)))
    print(wasserstein_dist(y_true.unsqueeze(0), y_input.unsqueeze(0)))
