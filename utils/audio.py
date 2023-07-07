import os
import librosa
import numpy as np
import pandas as pd
import pickle as pkl 
import torch
import torchaudio
from tqdm.auto import tqdm
from PIL import Image

from utils.transform import Log2Transform, MonoTransform, NormalizeTransform, TrimTransform

def spectrogram_split(spec, spec_win, spec_hop, spec_resize=-1):
    y, x = spec.shape
    slice = []
    n_frame = spec.shape[1]
    for i in range(len(spec_win)):
        win = spec_win[i]
        hop = spec_hop[i]
        # n_slice = (n_frame - win) // hop + 1
        # remain  = n_frame - (n_slice * hop - hop + win)
        # print(f'spec[{i}] win:{win} hop:{hop} n_slice:{n_slice} remain:{remain}')

        start = 0
        end = 0
        while end < n_frame:
            end = min(start + win, n_frame)
            # merge next frame if next frame < win // 2
            if n_frame - start - hop < win // 2:
                end = n_frame
            
            # print(f'  slice = [{start} - {end}] = {end - start}')
            img = Image.fromarray(spec[:, start:end] * 255)
            if spec_resize != end - start:
                img = img.resize((spec_resize, y))
            slice.append(np.array(img))
            start += hop

    return slice

def wav_remove_silent(waveform, top_db=30, frame_length=2048, hop_length=512):
    non_silent_intervals = librosa.effects.split(waveform, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    return np.concatenate([waveform[start:end] for start, end in non_silent_intervals])

def wav_trim(waveform, top_db=30, frame_length=2048, hop_length=512):
    return librosa.effects.trim(waveform, top_db=top_db, frame_length=frame_length, hop_length=hop_length)

def wav_to_spectrogram(wav, n_fft, win_length, hop_length, n_mels=128):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    preprocess = torch.nn.Sequential(
        mel_spectrogram,
        Log2Transform(),
        NormalizeTransform(),
    )

    return preprocess(wav)
