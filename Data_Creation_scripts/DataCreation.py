# ============================
# 📦 Imports
# ============================
import os
import re
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from typing import Tuple, List, Set
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Hebrew-related libraries
from hebrew import Hebrew
from hebrew.chars import HebrewChar, ALEPH
from hebrew import GematriaTypes
from HebrewToEnglish import HebrewToEnglish


MAX_DURATION = 15  # Maximum duration in seconds for audio files
# ============================
# Audio Utility
# ============================
def get_duration_in_seconds(audio_path: str) -> float:
    info = sf.info(audio_path)
    return info.frames / info.samplerate

# ============================
# Transcript Cleaning
# ============================
def filter_and_clean_transcripts(metadata: pd.DataFrame, audio_base_path: str) -> Tuple[pd.DataFrame, List[str]]:
    raw_transcript = metadata[["file_id", "transcript_in_english"]].copy()
    raw_transcript["transcript_in_english"] = raw_transcript["transcript_in_english"].astype(str).str.rstrip()

    word_counts = raw_transcript["transcript_in_english"].str.split().apply(len)
    too_few_words_mask = word_counts < 1
    durations = raw_transcript["file_id"].apply(lambda fid: get_duration_in_seconds(f"{audio_base_path}/{fid}.wav"))
    too_long_audio_mask = durations > MAX_DURATION

    bad_mask = too_few_words_mask | too_long_audio_mask
    notgood = raw_transcript.loc[bad_mask, "file_id"].tolist()

    print(f"Filtered out {len(notgood)} entries with too few words or too long audio.")
    print(f"Total entries before filtering: {len(raw_transcript)}")
    print(f"Total entries after filtering: {len(raw_transcript) - len(notgood)}")

    raw_transcript = raw_transcript.loc[~bad_mask].reset_index(drop=True)
    return raw_transcript, notgood

# ============================
# Audio Processing
# ============================
def process_audio_files(input_path: str, output_path: str, raw_transcript: pd.DataFrame, notgood: Set[str]) -> pd.DataFrame:
    os.makedirs(output_path, exist_ok=True)
    notgood = set(notgood)

    file_ids = []
    transcripts = []

    for filename in tqdm(os.listdir(input_path)):
        if not filename.endswith(".wav"):
            continue

        full_path = os.path.join(input_path, filename)
        file_id = filename[:-4]

        if file_id in raw_transcript["file_id"].values and get_duration_in_seconds(full_path) >= 0:
            y, sr = librosa.load(full_path, sr=22050)
            sf.write(os.path.join(output_path, filename), y, samplerate=22050)

            row = raw_transcript.loc[raw_transcript["file_id"] == file_id].iloc[0]
            file_ids.append(row["file_id"])
            transcripts.append(row["transcript_in_english"])

    return pd.DataFrame({"file_id": file_ids, "transcript_in_english": transcripts})

# ============================
# Dataset Configuration
# ============================
def get_dataset_config(set_type: str, config: dict) -> Tuple[pd.DataFrame, str, str, str]:
    if set_type not in config:
        raise ValueError(f"Unsupported set_type '{set_type}'. Choose from {list(config.keys())}.")

    cfg = config[set_type]
    metadata = pd.read_csv(cfg["metadata_path"], delimiter='|')
    return metadata, cfg["file_name_output"], cfg["input_path"], cfg["output_path"]

# ============================
# Transcript Writer
# ============================
def write_transcript_file_mels(raw_transcript: pd.DataFrame, file_name_output: str, input_path: str) -> None:
    with open(file_name_output, 'w', encoding='utf-8') as file:
        for row in raw_transcript.itertuples(index=False):
            mel_path = os.path.join(input_path, f"{row.file_id}.npy")
            text = row.transcript_in_english.rstrip()
            if not text.endswith('.'):
                text += '.'
            file.write(f"{mel_path}|{text}\n")

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    config = {
        "train": {
            "metadata_path": f"/gpfs0/bgu-benshimo/users/wavishay/projects/roboshual/saspeech_automatic_data/metadata.csv",
            "input_path": f"/gpfs0/bgu-benshimo/users/wavishay/projects/roboshual/saspeech_automatic_data/wavs/",
            "output_path": f"/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/data/trimm/saspeech_automatic_data/wavs/",
            "file_name_output": f"/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/filelists/train_list{MAX_DURATION}.txt"
        },
        "dev": {
            "metadata_path": f"/gpfs0/bgu-benshimo/users/wavishay/projects/roboshual/saspeech_gold_standard/metadata_full.csv",
            "input_path": f"/gpfs0/bgu-benshimo/users/wavishay/projects/roboshual/saspeech_gold_standard/wavs/",
            "output_path": f"/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/data/trimm/saspeech_gold_standard/wavs/",
            "file_name_output": f"/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/filelists/dev_list{MAX_DURATION}.txt"
        }
    }

    for set_type in ['train', 'dev']:
        metadata, file_name_output, input_path, output_path = get_dataset_config(set_type, config)

        metadata["transcript_in_english"] = metadata["transcript"].apply(HebrewToEnglish)
        raw_transcript, notgood = filter_and_clean_transcripts(metadata, input_path)

        # processed_transcripts = process_audio_files(
        #     input_path=input_path,
        #     output_path=output_path,
        #     raw_transcript=raw_transcript,
        #     notgood=notgood
        # )

        mel_output_path = output_path.replace("wavs", "mel_spectrograms")
        write_transcript_file_mels(raw_transcript, file_name_output, mel_output_path)

        print(f"✓ Completed processing for {set_type} set.")
