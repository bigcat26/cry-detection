import argparse
import librosa
import tqdm
import pandas as pd
import pickle as pkl

# add parent directory to import path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import *

# esc50_config = {
#     'num_folds': 1,
#     'n_fft': 512,
#     'hop_length': 256,
#     'win_length': 512,
#     'n_mels': 128,
#     'spec_win': [192, 256, 320],
#     'spec_hop': [96, 128, 160],
#     'spec_resize': 256,
#     'csv_file': 'E:/dataset/ESC-50-master/meta/esc50.csv',
#     'data_dir': 'E:/dataset/ESC-50-master/audio',
#     'store_dir': 'E:/dataset/out/esc50pp',
#     # 'csv_file': '../data/ESC-50-master/meta/esc50.csv',
#     # 'data_dir': '../data/ESC-50-master/audio',
#     # 'store_dir': './dataset',
#     'sampling_rate': 24000
# }

def extract_features_esc50(config, audios):
    audio_names = list(audios.filename.unique())
    values = []
    for file_name in tqdm(audio_names):
        full_path = f"{config['data_dir']}/{file_name}"
        clip, _ = librosa.load(full_path, sr=config['sampling_rate'])
        clip, _ = wav_trim(clip)
        clip = wav_remove_silent(clip)
        spec = wav_to_spectrogram(torch.Tensor(clip), config['n_fft'], config['win_length'], config['hop_length'], config['n_mels'])
        slices = spectrogram_split(spec.numpy(), config['spec_win'], config['spec_hop'], config['spec_resize'])
        entries = audios.loc[audios["filename"]==file_name].to_dict(orient="records")
        target = entries[0]['target']
        for v in slices:
            values.append({'value': v, 'target': target})
        # print(f"Finished {file_name} (target:{target})")
    return values

def preprocess_esc50(config):
    audios = pd.read_csv(config['csv_file'], skipinitialspace=True)
    num_folds = config['num_folds']

    if not os.path.exists(config['store_dir']):
        print(f"creating directory: {config['store_dir']}")
        os.makedirs(config['store_dir'])

    if num_folds == 0:
        # all in one
        print(f'Dumping all data to {config["store_dir"]}/all.pkl')
        values = extract_features_esc50(config, audios)
        with open(f"{config['store_dir']}/all.pkl", "wb") as handler:
            pkl.dump(values, handler, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        for i in range(1, num_folds+1):
            training_audios = audios.loc[audios["fold"]!=i]
            validation_audios = audios.loc[audios["fold"]==i]

            print(f'Dumping training data to {config["store_dir"]}/training{i}.pkl')
            training_values = extract_features_esc50(config, training_audios)
            with open(f"{config['store_dir']}/training{i}.pkl", "wb") as handler:
                pkl.dump(training_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

            print(f'Dumping validation data to {config["store_dir"]}/validation{i}.pkl')
            validation_values = extract_features_esc50(config, validation_audios)
            with open(f"{config['store_dir']}/validation{i}.pkl", "wb") as handler:
                pkl.dump(validation_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

def print_config(config):
    print(f"num_folds: {config['num_folds']}")
    print(f"n_fft: {config['n_fft']}")
    print(f"hop_length: {config['hop_length']}")
    print(f"win_length: {config['win_length']}")
    print(f"n_mels: {config['n_mels']}")
    print(f"spec_win: {config['spec_win']}")
    print(f"spec_hop: {config['spec_hop']}")
    print(f"spec_resize: {config['spec_resize']}")
    print(f"csv_file: {config['csv_file']}")
    print(f"data_dir: {config['data_dir']}")
    print(f"store_dir: {config['store_dir']}")
    print(f"sampling_rate: {config['sampling_rate']}")

# for a 24kHz sampling rate audio, generate spectrogram with 400 win_length and 200 hop length
# there are 120 fft frames per second (24000/200)
# by default, we use 1.5s, 2s, 2.5s window sizes, which are 180, 240, 300 frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess ESC-50 dataset')
    parser.add_argument('--num_folds', type=int, default=0, help='number of folds')
    parser.add_argument('--n_fft', type=int, default=400, help='number of fft points')
    parser.add_argument('--hop_length', type=int, default=200, help='hop length')
    parser.add_argument('--win_length', type=int, default=400, help='window length')
    parser.add_argument('--n_mels', type=int, default=128, help='number of mel filters')
    parser.add_argument('--spec_win', type=int, nargs='+', default=[180, 240, 300], help='spectrogram window sizes')
    parser.add_argument('--spec_hop', type=int, nargs='+', default=[60, 60, 60], help='spectrogram hop sizes')
    parser.add_argument('--spec_resize', type=int, default=256, help='spectrogram resize')
    parser.add_argument('--csv_file', type=str, default='E:/dataset/ESC-50-master/meta/esc50.csv', help='csv file')
    parser.add_argument('--data_dir', type=str, default='E:/dataset/ESC-50-master/audio', help='data directory')
    parser.add_argument('--store_dir', type=str, default='E:/dataset/out/esc50pp', help='store directory')
    parser.add_argument('--sampling_rate', type=int, default=24000, help='sampling rate')
    args = parser.parse_args()

    config = {
        'num_folds': args.num_folds,
        'n_fft': args.n_fft,
        'hop_length': args.hop_length,
        'win_length': args.win_length,
        'n_mels': args.n_mels,
        'spec_win': args.spec_win,
        'spec_hop': args.spec_hop,
        'spec_resize': args.spec_resize,
        'csv_file': args.csv_file,
        'data_dir': args.data_dir,
        'store_dir': args.store_dir,
        'sampling_rate': args.sampling_rate
    }
    
    print_config(config)
    preprocess_esc50(config)
