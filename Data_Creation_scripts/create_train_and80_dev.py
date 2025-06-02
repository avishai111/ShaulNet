import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# -----------------------------
# File Paths
# -----------------------------

MAX_DURATION = 12 

# Existing file lists
train_list_path = f"/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/filelists/train_list{MAX_DURATION}.txt"
dev_list_path = f"/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/filelists/dev_list{MAX_DURATION}.txt"
dev_20_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/filelists/dev_20dev_list.txt"

# Output file lists
weighted_train_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/filelists/train_with_80percent_dev_weighted.txt"
new_dev_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/filelists/dev_with_20percent_weighted.txt"

# -----------------------------
# Load Function
# -----------------------------
def load_transcript_list(file_path: str) -> pd.DataFrame:
    """Load a transcript file into a DataFrame with columns: mel_path, transcript"""
    df = pd.read_csv(file_path, sep='|', names=['mel_path', 'transcript'], engine='python')
    df['mel_path'] = df['mel_path'].str.strip()  # Clean potential spaces
    return df

# -----------------------------
# Load Data
# -----------------------------
train_df = load_transcript_list(train_list_path)
dev_df = load_transcript_list(dev_list_path)
dev_20_df = load_transcript_list(dev_20_path)

print("train_df:", train_df.isnull().any())
print("dev_df:", dev_df.isnull().any())
print("dev_20_df:", dev_20_df.isnull().any())
# -----------------------------
# Create New Dev Set (20%)
# -----------------------------
dev_df['mel_path'] = dev_df['mel_path'].str.strip()
dev_20_df['mel_path'] = dev_20_df['mel_path'].str.strip()

# סנן את dev_df לפי קבצים שקיימים ב-dev_20_df
new_dev_df = dev_df[dev_df['mel_path'].isin(dev_20_df['mel_path'])].copy()

print("new_dev_df:", new_dev_df.isnull().any())
# Validate all entries in dev_20 exist in dev
missing = new_dev_df[new_dev_df['transcript'].isnull()]
if not missing.empty:
    print("⚠️ WARNING: Missing transcripts for some dev_20 files:")
    print(missing)
    raise ValueError("Some files in dev_20 not found")

# -----------------------------
#  Create Weighted Training Set (Train + 80% Dev)
# -----------------------------
remaining_dev_df = dev_df[~dev_df['mel_path'].isin(dev_20_df['mel_path'])]

num_train = len(train_df)
num_dev = len(remaining_dev_df)
total = num_train + num_dev

train_weight = total / num_train
dev_weight = total / num_dev

#scaling_factor = len(train_df) / len(remaining_dev_df)

train_df["weight"] = 1.0
remaining_dev_df["weight"] = 2.0

weighted_train_df = pd.concat([train_df, remaining_dev_df]).reset_index(drop=True)

# -----------------------------
#  Save New File Lists
# -----------------------------
weighted_train_df.to_csv(weighted_train_path, sep='|', index=False, header=False)
new_dev_df[['mel_path', 'transcript']].to_csv(new_dev_path, sep='|', index=False, header=False)

# -----------------------------
# ✅ Summary
# -----------------------------
print("✓ Created weighted training and dev files:")
print(f" - Weighted train: {weighted_train_path}")
print(f" - 20% Dev set:    {new_dev_path}")
