import csv
import math
import os
import pathlib
import random
from glob import glob
from pathlib import Path

import librosa

import numpy as np
import soundfile as sf

import pyloudnorm as pyln

def collect_sub_video_folders(folder_root: str):
    sub_video_folders = []

    # Walk through the folder structure
    for root, dirs, files in os.walk(folder_root):
        # Look for "sub_video" in the path
        if "sub_video" in root:
            # If we find a folder that is one level deeper than "sub_video"
            # and stop at that level
            for dir_name in dirs:
                sub_video_path = os.path.join(root, dir_name)
                sub_video_folders.append(sub_video_path)

            # Prevent further walking inside the folders found inside "sub_video"
            dirs.clear()

    return sub_video_folders

# measure the loudness first 
meter = pyln.Meter(44100) # create BS.1770 meter

def measure_loudness(samples, sr=44100):
    return meter.integrated_loudness(samples)

def adjust_loudness(samples, target_loudness, sr=44100):
    current_loudness = measure_loudness(samples, sr)
    print(f"Current loudness: {current_loudness:.2f} dB")

    change_in_dB = target_loudness - current_loudness
    print(f"Change in dB: {change_in_dB:.2f}")

    loudness_normalized_audio = pyln.normalize.loudness(samples, current_loudness, target_loudness)

    return loudness_normalized_audio

def lufs_norm(data, sr, norm=-6):
    block_size = 0.4 if len(data) / sr >= 0.4 else len(data) / sr
    # measure the loudness first
    meter = pyln.Meter(rate=sr, block_size=block_size)
    loudness = meter.integrated_loudness(data)

    assert not math.isinf(loudness)

    norm_data = pyln.normalize.loudness(data, loudness, loudness + norm)
    n, d = np.sum(np.array(norm_data)), np.sum(np.array(data))
    gain = n / d if d else 0.0

    return norm_data, gain


def get_lufs(data, sr):
    block_size = 0.4 if len(data) / sr >= 0.4 else len(data) / sr
    # measure the loudness first
    meter = pyln.Meter(rate=sr, block_size=block_size)  # create BS.1770 meter
    loudness = meter.integrated_loudness(data)

    return loudness


def peak_norm(data, mx):
    eps = 1e-10
    max_sample = np.max(np.abs(data))
    scale_factor = mx / (max_sample + eps)

    return data * scale_factor


def gain_to_db(g):
    return 20 * np.log10(g)


def db_to_gain(db):
    return 10 ** (db / 20.0)


suppress_dB = [-12, -9, -6,]
highlight_dB = [6, 9, 12]
categories = ["speech", "music", "sfx"]


class MixAudio:
    def __init__(self, peak_norm_db=-0.5):
        self.root = "./data/Muddy_Mix"
        self.folder_paths = collect_sub_video_folders(self.root)
        print(len(self.folder_paths))
        self.peak_norm = db_to_gain(peak_norm_db)
        self.csv_rows = []
        self.mixaudio()

    def load_audio(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=44100)
        return audio, sr

    def sample_loudness_change(self, num_sup=2, num_aug=1):
        sup_loudnesses = random.choices(suppress_dB, k=int(num_sup))
        aug_loudnesses = random.choices(highlight_dB, k=int(num_aug))
        return sup_loudnesses[0], sup_loudnesses[1], aug_loudnesses[0]

    def sample_highlighting_class(self):
        return random.choices(categories, k=1)[0]

    def output_csv(self, output_path):
        csv_path = os.path.join(output_path, "metadata_onedb.csv")
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "folder_path",
                    "speech_loudness",
                    "music_loudness",
                    "sfx_loudness",
                    "highlight_class",
                    "speech_loudness_change",
                    "music_loudness_change",
                    "sfx_loudness_change",
                    "target_mix",
                    "gain_speech",
                    "gain_music",
                    "gain_sfx",
                ]
            )
            writer.writerows(self.csv_rows)

    def mixaudio(self):
        for folder_path in self.folder_paths:
            # mixture audio is in folder_path/audio_raw/*.wav
            mixture_audio_path = glob(folder_path + "/audio_raw/*.wav")[0]
            # if no "separated" or no corresponding wav files, skip
            speech_audio_path = os.path.join(folder_path, "separated", "speech.wav")
            music_audio_path = os.path.join(folder_path, "separated", "music.wav")
            sfx_audio_path = os.path.join(folder_path, "separated", "effects+residual.wav") # we are using the effects+residual as sfx
            if not os.path.exists(speech_audio_path) or not os.path.exists(music_audio_path) or not os.path.exists(sfx_audio_path):
                print(f"Skipping folder {folder_path} due to missing audio files")
                continue
            else:
                # Continue processing the audio files
                print(f"Processing audio in {folder_path}")

            # define output path
            output_path = os.path.join(folder_path, "remix_global_onedb")
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Check if remix_global exists and contains any .wav files
            if os.path.exists(output_path) and any(f.endswith(".wav") for f in os.listdir(output_path)):
                print(f"Skipping {output_path} because it already contains .wav files")
                continue

            mixture_audio, sr = self.load_audio(mixture_audio_path)
            speech_audio, sr = self.load_audio(speech_audio_path)
            music_audio, sr = self.load_audio(music_audio_path)
            sfx_audio, sr = self.load_audio(sfx_audio_path)

            # the original audio is around 10 seconds
            mixture_segment = mixture_audio
            speech_segment = speech_audio
            music_segment = music_audio
            sfx_segment = sfx_audio

            # save copy of original audio
            original_speech_segment = speech_segment.copy()
            original_music_segment = music_segment.copy()
            original_sfx_segment = sfx_segment.copy()

            # get original loudness
            speech_loudness = get_lufs(speech_segment, sr)
            music_loudness = get_lufs(music_segment, sr)
            sfx_loudness = get_lufs(sfx_segment, sr)
            
            # assign silence flag 
            silence_flag_speech = False
            silence_flag_music = False
            silence_flag_sfx = False
            if speech_loudness == -np.inf:
                silence_flag_speech = True
            if music_loudness == -np.inf:
                silence_flag_music = True
            if sfx_loudness == -np.inf:
                silence_flag_sfx = True

            # find the loudest sound category, speech, music or sfx
            max_audio_class = None
            max_loudness = max(speech_loudness, music_loudness, sfx_loudness)
            if max_loudness == speech_loudness:
                max_audio_class = "speech"
            elif max_loudness == music_loudness:
                max_audio_class = "music"
            else:
                max_audio_class = "sfx"

            mix_max_peak = 0.0
            target_mix = np.zeros_like(mixture_segment)

            highlight_class = self.sample_highlighting_class()
            # sample highlighting class
            while highlight_class == max_audio_class:
                highlight_class = self.sample_highlighting_class()

            loudness_changes = self.sample_loudness_change()
            # shift loundess_changes to suppress the loudest sound category, and highlight the selected sound category, and suppress the other
            loudness = [0, 0, 0]
            loudness[["speech", "music", "sfx"].index(max_audio_class)] = loudness_changes[0]
            loudness[["speech", "music", "sfx"].index(highlight_class)] = loudness_changes[2]
            loudness[["speech", "music", "sfx"].index(list(set(["speech", "music", "sfx"]) - set([max_audio_class, highlight_class]))[0])] = loudness_changes[1]
            
            # print(folder_path)
            # print(f"loudness: {loudness}")
            # print(f"highlight_class: {highlight_class}")
            # print(f"max_audio_class: {max_audio_class}")
            # print(f"loudness_changes: {loudness_changes}")
            # print(f"speech_loudness: {speech_loudness}, music_loudness: {music_loudness}, sfx_loudness: {sfx_loudness}")
            # exit()

            # adjust loudness
            try:
                if not silence_flag_speech:
                    speech_segment_, gain_speech = lufs_norm(
                        speech_segment, sr, norm=loudness[0]
                    )
                    mix_max_peak = max(
                        mix_max_peak, np.max(np.abs(speech_segment_))
                    )
                else:
                    speech_segment_ = speech_segment.copy()
                    gain_speech = 0.0
                    print("Silence detected in speech segment")
            except AssertionError:
                speech_segment_ = speech_segment.copy()
                gain_speech = 0.0
                print("Silence detected in speech segment")
            try:
                if not silence_flag_music:
                    music_segment_, gain_music = lufs_norm(
                        music_segment, sr, norm=loudness[1]
                    )
                    mix_max_peak = max(mix_max_peak, np.max(np.abs(music_segment_)))
                else:
                    music_segment_ = music_segment.copy()
                    gain_music = 0.0
                    print("Silence detected in music segment")
            except AssertionError:
                music_segment_ = music_segment.copy()
                gain_music = 0.0
                print("Silence detected in music segment")
            try:
                if not silence_flag_sfx:
                    sfx_segment_, gain_sfx = lufs_norm(
                        sfx_segment, sr, norm=loudness[2]
                    )
                    mix_max_peak = max(mix_max_peak, np.max(np.abs(sfx_segment_)))
                else:
                    sfx_segment_ = sfx_segment.copy()
                    gain_sfx = 0.0
                    print("Silence detected in sfx segment")
            except AssertionError:
                sfx_segment_ = sfx_segment.copy()
                gain_sfx = 0.0
                print("Silence detected in sfx segment")

            # peak norm
            peak_norm_gain = (
                1.0
                if mix_max_peak <= self.peak_norm
                else self.peak_norm / mix_max_peak
            )
            speech_segment_ *= peak_norm_gain
            music_segment_ *= peak_norm_gain
            sfx_segment_ *= peak_norm_gain
            gain_speech *= peak_norm_gain
            gain_music *= peak_norm_gain
            gain_sfx *= peak_norm_gain

            # mix audio
            target_mix = speech_segment_ + music_segment_ + sfx_segment_

            # detect if targe_mix is too loud
            target_mix_max_peak = np.max(np.abs(target_mix))
            if target_mix_max_peak > self.peak_norm:
                target_mix *= self.peak_norm / target_mix_max_peak
                speech_segment_ *= self.peak_norm / target_mix_max_peak
                music_segment_ *= self.peak_norm / target_mix_max_peak
                sfx_segment_ *= self.peak_norm / target_mix_max_peak
                gain_speech *= self.peak_norm / target_mix_max_peak
                gain_music *= self.peak_norm / target_mix_max_peak
                gain_sfx *= self.peak_norm / target_mix_max_peak

            csv_row = [
                folder_path,
                speech_loudness,
                music_loudness,
                sfx_loudness,
                highlight_class,
                loudness[0],
                loudness[1],
                loudness[2],
                target_mix,
                gain_speech,
                gain_music,
                gain_sfx,
            ]

            self.csv_rows.append(csv_row)

            # save audio
            sf.write(os.path.join(output_path, "target_mix.wav"), target_mix, sr)
            sf.write(os.path.join(output_path, "speech.wav"), speech_segment_, sr)
            sf.write(os.path.join(output_path, "music.wav"), music_segment_, sr)
            sf.write(os.path.join(output_path, "effects.wav"), sfx_segment_, sr)
        self.output_csv("./data/new_data")

mix = MixAudio(peak_norm_db=-0.5)