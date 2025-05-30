import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
# קבצים קיימים
train_list_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/filelists/train_list30.txt"
dev_list_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/filelists/dev_list_mels.txt"
dev_20_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/filelists/dev_20dev_list.txt"

# קבצים חדשים ליצוא
weighted_train_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/train_with_80percent_dev_weighted.txt"
new_dev_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/dev_20percent_list_weighted.txt"

# פונקציה לטעינה
def load_transcript_list(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep='|', names=['mel_path', 'transcript'], engine='python')

# טען את כל הקבצים
train_df = load_transcript_list(train_list_path)
dev_df = load_transcript_list(dev_list_path)
dev_20_df = load_transcript_list(dev_20_path)

# נקה רווחים מיותרים (אם קיימים)
train_df['mel_path'] = train_df['mel_path'].str.strip()
dev_df['mel_path'] = dev_df['mel_path'].str.strip()
dev_20_df['mel_path'] = dev_20_df['mel_path'].str.strip()

# סט הולידציה מדויק לפי הסדר של dev_20_df
new_dev_df = pd.merge(dev_20_df[['mel_path']], dev_df, on='mel_path', how='left')

# בדיקה – האם הכל נמצא?
missing = new_dev_df[new_dev_df['transcript'].isnull()]
if not missing.empty:
    print("⚠️ WARNING: Missing transcripts for some dev_20 files:")
    print(missing)
    raise ValueError("Some files in dev_20 not found in dev_list.txt")

# שאר קבצי dev — עבור אימון
remaining_dev_df = dev_df[~dev_df['mel_path'].isin(dev_20_df['mel_path'])]

# חישוב משקלים יחסיים
num_train = len(train_df)
num_dev = len(remaining_dev_df)
total = num_train + num_dev

train_weight = total / num_train
dev_weight = total / num_dev

# הוסף עמודת משקל
train_df["weight"] = train_weight
remaining_dev_df["weight"] = dev_weight

# איחוד
weighted_train_df = pd.concat([train_df, remaining_dev_df]).reset_index(drop=True)

# שמור את קובץ האימון החדש עם משקלים
weighted_train_df.to_csv(weighted_train_path, sep='|', index=False, header=False)

# שמור את קובץ הולידציה (בסדר מדויק)
new_dev_df[['mel_path', 'transcript']].to_csv(new_dev_path, sep='|', index=False, header=False)

print("✓ Created weighted training file:")
print(f"- Weighted train: {weighted_train_path}")
print(f"- Dev: {new_dev_path}")


# def get_mel_durations(df: pd.DataFrame, base_dir="", hop_length=256, sampling_rate=22050) -> list:
#     durations = []
#     for path in df['mel_path']:
#         full_path = os.path.join(base_dir, path)
#         try:
#             if full_path.endswith('.npy'):
#                 mel = np.load(full_path)
#             else:
#                 mel = torch.load(full_path)
#             num_frames = mel.shape[-1]
#             duration_sec = num_frames * hop_length / sampling_rate
#             durations.append(duration_sec)
#         except Exception as e:
#             print(f"❌ Failed to load {full_path}: {e}")
#     return durations



# print("📊 Calculating mel lengths for histogram...")
# all_lengths = get_mel_durations(pd.concat([train_df, dev_df]))

# plt.figure(figsize=(10, 5))
# plt.hist(all_lengths, bins=50, color='gray', edgecolor='black')
# plt.title("Histogram of Mel Spectrogram Lengths")
# plt.xlabel("Number of Frames")
# plt.ylabel("Count")
# plt.grid(True)
# plt.tight_layout()

# # שמירה לקובץ
# histogram_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/mel_length_histogram.png"
# plt.savefig(histogram_path, dpi=300)
# print(f"📁 Histogram saved to: {histogram_path}")

# plt.close()