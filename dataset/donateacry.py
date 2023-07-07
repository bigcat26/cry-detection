

import os

def find_wav_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.wav'):
                filepath = os.path.join(dirpath, filename)
                print(filepath)