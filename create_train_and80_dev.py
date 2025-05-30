import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
# exsiting path files
train_list_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/filelists/train_list30.txt"
dev_list_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/filelists/dev_list_mels.txt"
dev_20_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/filelists/dev_20dev_list.txt"

# new path files
weighted_train_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/train_with_80percent_dev_weighted.txt"
new_dev_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/dev_20percent_list_weighted.txt"

# load transcript list function
def load_transcript_list(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep='|', names=['mel_path', 'transcript'], engine='python')

# loading the path files into dataframes
train_df = load_transcript_list(train_list_path)
dev_df = load_transcript_list(dev_list_path)
dev_20_df = load_transcript_list(dev_20_path)

# Cleaning spaces in mel_path if they exist
train_df['mel_path'] = train_df['mel_path'].str.strip()
dev_df['mel_path'] = dev_df['mel_path'].str.strip()
dev_20_df['mel_path'] = dev_20_df['mel_path'].str.strip()

# create 20% from dev, the same as dev_20_df for consistency and comparison the models.
new_dev_df = pd.merge(dev_20_df[['mel_path']], dev_df, on='mel_path', how='left')

# Check if all mel_paths in dev_20_df exist in dev_df
missing = new_dev_df[new_dev_df['transcript'].isnull()]
if not missing.empty:
    print("⚠️ WARNING: Missing transcripts for some dev_20 files:")
    print(missing)
    raise ValueError("Some files in dev_20 not found in dev_list.txt")

# creating new training set with 80% of dev.
remaining_dev_df = dev_df[~dev_df['mel_path'].isin(dev_20_df['mel_path'])]

# calculate weights for training and dev sets.
num_train = len(train_df)
num_dev = len(remaining_dev_df)
total = num_train + num_dev

train_weight = total / num_train
dev_weight = total / num_dev

# adding weights to the dataframes.
train_df["weight"] = train_weight
remaining_dev_df["weight"] = dev_weight

# Concatenate the training set with the remaining dev set to create a new weighted training set
weighted_train_df = pd.concat([train_df, remaining_dev_df]).reset_index(drop=True)

# Save the new files with the weights included.
weighted_train_df.to_csv(weighted_train_path, sep='|', index=False, header=False)

# Save the new dev set with 20% from the original dev set
new_dev_df[['mel_path', 'transcript']].to_csv(new_dev_path, sep='|', index=False, header=False)

print("✓ Created weighted training file:")
print(f"- Weighted train: {weighted_train_path}")
print(f"- Dev: {new_dev_path}")


