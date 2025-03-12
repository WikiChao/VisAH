import os
import csv
import torchaudio
import torch
from pathlib import Path
import pandas as pd

# dataset class for loading movie data
# define dataset loader
class BaseAudioDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path="./data",mode="val", sample_rate=44100, num_time_frames=10, num_samples=None):
        # Define the root directory for the dataset
        if mode == "train":
            self.csv_file = os.path.join(metadata_path, "train.csv")
        elif mode == "val":
            self.csv_file = os.path.join(metadata_path, "val.csv")
        elif mode == "test":   
            self.csv_file = os.path.join(metadata_path, "test_filter.csv")
        else:
            raise ValueError("mode should be train, val or test")

        self.mode = mode

        self.metadata_csv = os.path.join(metadata_path, "metadata_onedb.csv")

        # Load the data from the CSV file with csv
        with open(self.csv_file, "r") as f:
            self.data = list(csv.reader(f))

        # Load the metadata from the CSV file
        self.metadata = pd.read_csv(self.metadata_csv)

        if num_samples is not None:
            self.data = self.data[:num_samples]

        self.sample_rate = sample_rate
        self.num_time_frames = num_time_frames

        self.audio_names = ["speech.wav", "music.wav", "effects+residual.wav"]

        print(f"Loaded {len(self.data)} samples")

    def load_audio(self, audio_path):
        # Load the audio file
        audio, sample_rate = torchaudio.load(audio_path)
        # Convert the audio to mono if it is stereo
        if audio.shape[0] == 2:
            audio = (audio[0:1, :] + audio[1:2, :]) / 2
            
        # check Nan and inf, if any, assert
        assert not torch.isnan(audio).any(), "Nan in audio"
        assert not torch.isinf(audio).any(), "Inf in audio"

        # Return the audio and sample rate
        return audio, sample_rate

    def resample_audio(self, audio, sample_rate):
        # Resample the audio to the target sample rate
        if sample_rate != self.sample_rate:
            audio = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(audio)
        return audio
    
    def pad_audio(self, audio):
        # Pad the audio to the target length
        if audio.size(1) < self.num_time_frames * self.sample_rate:
            padding = torch.zeros(1, self.num_time_frames * self.sample_rate - audio.size(1))
            audio = torch.cat([audio, padding], dim=1)
        return audio

    def sample_time_frames_audio(self, audio):
        # if the audio is longer than num_time_frames, select the first num_time_frames seconds
        if audio.size(1) > self.num_time_frames * self.sample_rate:
            audio = audio[:, :self.num_time_frames * self.sample_rate]
        return audio
    
    def load_all_audios(self, folder_path):
        # load all audios with name in self.audio_names, and use dict to store them. simplify the key with only the prefix
        audios = {}
        for audio_name in self.audio_names:
            audio_path = os.path.join(folder_path, audio_name)
            audio, sample_rate = self.load_audio(audio_path)
            audios[audio_name.split(".")[0]] = audio
        return audios
    
    def load_audio_info(self, folder_path):
        # find the row in the metadata csv file
        row = self.metadata[self.metadata["folder_path"] == folder_path]
        # get the loudness 
        # check if the row is empty
        if row.empty:
            speech_loudness = float("-inf")
            music_loudness = float("-inf")
            effects_loudness = float("-inf")
            return speech_loudness, music_loudness, effects_loudness
        speech_loudness = row["speech_loudness"].values[0]
        music_loudness = row["music_loudness"].values[0]
        effects_loudness = row["sfx_loudness"].values[0]
        return speech_loudness, music_loudness, effects_loudness

    def remix_on_the_fly(self, audios, speech_loudness, music_loudness, effects_loudness):
        # find the loudest audio category (-inf may exist)
        loudness = [speech_loudness, music_loudness, effects_loudness]
        loudness = [loud if loud != float("-inf") else -100 for loud in loudness]
        loudness = torch.tensor(loudness)
        max_loudness_idx = torch.argmax(loudness)
        # for the loudest audio category, apply scaling randomly from [0.1 to 0.7], and for the other two, apply scaling randomly from [1.5 to 3]
        for idx, audio_name in enumerate(self.audio_names):
            if idx == max_loudness_idx:
                scaling = torch.rand(1) * 0.6 + 0.1
            else:
                scaling = torch.rand(1) * 1.5 + 1.5
            audios[audio_name.split(".")[0]] = audios[audio_name.split(".")[0]] * scaling
        # sum the audios
        audio = sum(audios.values())
        # normalize the audio between [-1, 1]
        audio = audio / torch.max(torch.abs(audio))
        return audio

    def load_data(self, folder_path):
        # load frame feats
        frame_feats_path = os.path.join(folder_path, "frames_feats", "visual_feats.pt")
        frame_feats = torch.load(frame_feats_path, weights_only=True)
        # sample the first num_time_frames seconds
        frame_feats = frame_feats[:self.num_time_frames]
        # change the required grad to False
        frame_feats = frame_feats.detach()
        frame_feats.requires_grad = False

        # load text feats
        text_feats_path = os.path.join(folder_path, "frames_captions", "InternVL2-8B_prompt1_feats.pt")
        text_feats = torch.load(text_feats_path, weights_only=True)
        # sample the first num_time_frames seconds
        text_feats = text_feats[:self.num_time_frames]
        # change the required grad to False
        text_feats = text_feats.detach()
        text_feats.requires_grad = False

        # load text
        text_path = os.path.join(folder_path, "frames_captions", "InternVL2-8B_prompt1.txt")
        with open(text_path, "r") as f:
            text = f.readlines()
        text = text[0]

        # load target audio in audio_raw, *.wav
        target_audio_path = list((Path(folder_path) / "audio_raw").glob("*.wav"))[0]
        target_audio, sample_rate = self.load_audio(target_audio_path)
        # resample the audio to the target sample rate
        target_audio = self.resample_audio(target_audio, sample_rate)
        # pad the audio to the target length
        target_audio = self.pad_audio(target_audio)
        # sample the first num_time_frames seconds
        target_audio = self.sample_time_frames_audio(target_audio)

        # load input audio in remix_global, target.wav
        input_audio_path = Path(folder_path) / "remix_global" / "target_mix.wav"
        input_audio, sample_rate = self.load_audio(input_audio_path)
        # resample the audio to the target sample rate
        input_audio = self.resample_audio(input_audio, sample_rate)
        # pad the audio to the target length
        input_audio = self.pad_audio(input_audio)
        # sample the first num_time_frames seconds
        input_audio = self.sample_time_frames_audio(input_audio)

        # if self.mode != "train":
        #     # load input audio in remix_global, target.wav
        #     input_audio_path = Path(folder_path) / "remix_global" / "target_mix.wav"
        #     input_audio, sample_rate = self.load_audio(input_audio_path)
        #     # resample the audio to the target sample rate
        #     input_audio = self.resample_audio(input_audio, sample_rate)
        #     # pad the audio to the target length
        #     input_audio = self.pad_audio(input_audio)
        #     # sample the first num_time_frames seconds
        #     input_audio = self.sample_time_frames_audio(input_audio)
        # else:
        #     # load speech, music, effects+residual from separated folder
        #     separated_folder = Path(folder_path) / "separated"
        #     audios = self.load_all_audios(separated_folder)
        #     # load the loudness info
        #     speech_loudness, music_loudness, effects_loudness = self.load_audio_info(folder_path)
        #     # remix on the fly
        #     input_audio = self.remix_on_the_fly(audios, speech_loudness, music_loudness, effects_loudness)
        #     # resample the audio to the target sample rate
        #     input_audio = self.resample_audio(input_audio, sample_rate)
        #     # pad the audio to the target length
        #     input_audio = self.pad_audio(input_audio)
        #     # sample the first num_time_frames seconds
        #     input_audio = self.sample_time_frames_audio(input_audio)

        return frame_feats, target_audio, input_audio, text_feats, text

    def __getitem__(self, index):
        # Get the folder path from the data
        folder_path = self.data[index][0]
        # Load the data from the folder
        frame_feats, target_audio, input_audio, text_feats, text = self.load_data(folder_path)

        assert not torch.isnan(frame_feats).any(), "Nan in frame_feats"
        assert not torch.isinf(frame_feats).any(), "Inf in frame_feats"
        assert target_audio.shape[1] == self.num_time_frames * self.sample_rate, f"target_audio.shape[1]: {target_audio.shape[1]}, self.num_time_frames * self.sample_rate: {self.num_time_frames * self.sample_rate}"
        assert input_audio.shape[1] == self.num_time_frames * self.sample_rate, f"input_audio.shape[1]: {input_audio.shape[1]}, self.num_time_frames * self.sample_rate: {self.num_time_frames * self.sample_rate}"

        # Return the data
        data = {
            "frame_feats": frame_feats,
            "target_audio": target_audio,
            "input_audio": input_audio,
            "folder_path": folder_path,
            "text_feats": text_feats,
            "text": text
        }

        return data

    def __len__(self):
        return len(self.data)
    
# # unit test
# dataset = BaseAudioDataset(mode="val")
# # define data loader
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
# # get a batch of data
# for batch in data_loader:
#     data = batch
#     break
# data = dataset[0]
# print(data["frame_feats"].shape, data["target_audio"].shape, data["input_audio"].shape)