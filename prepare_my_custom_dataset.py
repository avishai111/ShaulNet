import os
import json
from pathlib import Path

def prepare_from_text_files(train_file, valid_file, save_folder):
    def parse_txt(file_path,wav_path = None):
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
        data = {}
        for line in lines:
            wav_path_file, label = line.strip().split("|", 1)
            uttid = Path(wav_path_file).stem
            data[uttid] = {
                "uttid": uttid,
                "wav": os.path.join(wav_path, wav_path_file + ".wav") if wav_path else wav_path_file + ".wav",
                "label": label.strip()
            }
        return data

    os.makedirs(save_folder, exist_ok=True)

    train_data = parse_txt(train_file, wav_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS/Text-To-speech/data/saspeech_automatic_data/")
    valid_data = parse_txt(valid_file, wav_path = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS/Text-To-speech/data/saspeech_gold_standard/")
    test_data = {}  # Optional

    with open(os.path.join(save_folder, "train.json"), "w") as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(save_folder, "valid.json"), "w") as f:
        json.dump(valid_data, f, indent=2)

    with open(os.path.join(save_folder, "test.json"), "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"Prepared JSONs: {len(train_data)} train, {len(valid_data)} valid.")


# import os
# import json
# import numpy as np
# import tgt
# import torchaudio
# from pathlib import Path
# from tqdm import tqdm

# def prepare_from_text_files_fastspeech2(
#     data_folder,
#     save_folder,
#     splits,
#     split_ratio,
#     model_name,
#     seed,
#     pitch_n_fft,
#     pitch_hop_length,
#     pitch_min_f0,
#     pitch_max_f0,
#     skip_prep,
#     use_custom_cleaner,
# ):
#     def parse_txt(file_path, wav_root):
#         with open(file_path, encoding="utf-8") as f:
#             lines = f.readlines()

#         data = {}
#         for line in lines:
#             wav_file, label = line.strip().split("|", 1)
#             uttid = Path(wav_file).stem
#             data[uttid] = {
#                 "uttid": uttid,
#                 "wav": os.path.join(wav_root, wav_file + ".wav"),
#                 "label": label.strip(),
#             }
#         return data

#     def enrich_entry(entry, alignment_dir, pitch_dir, duration_dir):
#         uttid = entry["uttid"]
#         wav = entry["wav"]

#         # Load audio
#         audio, fs = torchaudio.load(wav)
#         duration_sec = audio.shape[1] / fs
#         entry["duration"] = duration_sec
#         entry["segment"] = True

#         # Load TextGrid
#         textgrid_path = os.path.join(alignment_dir, uttid + ".TextGrid")
#         if not os.path.isfile(textgrid_path):
#             print(f"⚠️ Warning: TextGrid not found for {uttid}, skipping...")
#             return None

#         tg = tgt.io.read_textgrid(textgrid_path, include_empty_intervals=True)
#         phones = tg.get_tier_by_name("phones")
#         words = tg.get_tier_by_name("words")

#         # Extract phonemes and durations
#         last_flags = get_last_phoneme_info(words, phones)
#         phonemes, durations, start, end, spn_flags = get_alignment(
#             phones, fs, pitch_hop_length, last_flags
#         )

#         if start >= end or len(phonemes) == 0:
#             print(f"⚠️ Warning: Invalid alignment for {uttid}, skipping...")
#             return None

#         # Save durations
#         duration_path = os.path.join(duration_dir, uttid + ".npy")
#         np.save(duration_path, durations)
#         entry["durations"] = duration_path

#         # Extract and save pitch
#         trimmed_audio = audio[:, int(start * fs):int(end * fs)]
#         pitch = torchaudio.functional.detect_pitch_frequency(
#             waveform=trimmed_audio,
#             sample_rate=fs,
#             frame_time=(pitch_hop_length / fs),
#             win_length=3,
#             freq_low=pitch_min_f0,
#             freq_high=pitch_max_f0,
#         ).squeeze(0)

#         pitch = torch.cat([pitch, pitch[-1].unsqueeze(0)])  # match duration length
#         pitch = pitch[:sum(durations)]

#         # Optional normalization (adjust values to your dataset)
#         pitch = (pitch - 256.17) / 328.32

#         pitch_path = os.path.join(pitch_dir, uttid + ".npy")
#         np.save(pitch_path, pitch)

#         # Final additions
#         entry["pitch"] = pitch_path
#         entry["label_phoneme"] = " ".join(phonemes)
#         entry["spn_labels"] = spn_flags
#         entry["start"] = start
#         entry["end"] = end

#         return entry

#     # Setup folders
#     os.makedirs(save_folder, exist_ok=True)
#     pitch_dir = os.path.join(save_folder, "pitch")
#     duration_dir = os.path.join(save_folder, "durations")
#     os.makedirs(pitch_dir, exist_ok=True)
#     os.makedirs(duration_dir, exist_ok=True)

#     # Define your .txt input files (hardcoded here, update as needed)
#     train_txt = os.path.join("/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS/Text-To-speech/", "train_list2_HebrewToEnglish.txt")
#     valid_txt = os.path.join("/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS/Text-To-speech/", "dev_list.txt")
#     train_wav_root = os.path.join("/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS/Text-To-speech/data", "saspeech_automatic_data")
#     valid_wav_root = os.path.join("/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS/Text-To-speech/data", "saspeech_gold_standard")
#     alignment_dir = os.path.join("/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS/Text-To-speech/data", "TextGrid")

#     train_data = parse_txt(train_txt, train_wav_root)
#     valid_data = parse_txt(valid_txt, valid_wav_root)

#     enriched_train = {}
#     print("🔧 Enriching training data...")
#     for key, entry in tqdm(train_data.items()):
#         enriched = enrich_entry(entry, alignment_dir, pitch_dir, duration_dir)
#         if enriched is not None:
#             enriched_train[key] = enriched

#     enriched_valid = {}
#     print("🔧 Enriching validation data...")
#     for key, entry in tqdm(valid_data.items()):
#         enriched = enrich_entry(entry, alignment_dir, pitch_dir, duration_dir)
#         if enriched is not None:
#             enriched_valid[key] = enriched

#     # Save JSONs
#     with open(os.path.join(save_folder, "train.json"), "w") as f:
#         json.dump(enriched_train, f, indent=2)

#     with open(os.path.join(save_folder, "valid.json"), "w") as f:
#         json.dump(enriched_valid, f, indent=2)

#     with open(os.path.join(save_folder, "test.json"), "w") as f:
#         json.dump({}, f, indent=2)

#     print(f"✅ Prepared: {len(enriched_train)} train, {len(enriched_valid)} valid")


# # Helper functions
# def get_alignment(tier, sampling_rate, hop_length, last_phoneme_flags):
#     sil_phones = ["sil", "sp", "spn", ""]
#     phonemes, durations = [], []
#     start_time, end_time, end_idx = 0, 0, 0
#     trimmed_last_phoneme_flags = []

#     flag_iter = iter(last_phoneme_flags)

#     for t in tier._objects:
#         s, e, p = t.start_time, t.end_time, t.text
#         current_flag = next(flag_iter)

#         if phonemes == [] and p in sil_phones:
#             continue
#         if phonemes == []:
#             start_time = s

#         if p not in sil_phones:
#             if p[-1].isdigit():
#                 phonemes.append(p[:-1])
#             else:
#                 phonemes.append(p)
#             trimmed_last_phoneme_flags.append(current_flag[1])
#             end_time = e
#             end_idx = len(phonemes)
#         else:
#             phonemes.append("spn")
#             trimmed_last_phoneme_flags.append(current_flag[1])

#         durations.append(
#             int(np.round(e * sampling_rate / hop_length) - np.round(s * sampling_rate / hop_length))
#         )

#     phonemes = phonemes[:end_idx]
#     durations = durations[:end_idx]
#     return phonemes, durations, start_time, end_time, trimmed_last_phoneme_flags


# def get_last_phoneme_info(words_seq, phones_seq):
#     phoneme_iter = iter(phones_seq._objects)
#     last_phoneme_flags = []
#     for word_obj in words_seq._objects:
#         word_end_time = word_obj.end_time
#         current_phoneme = next(phoneme_iter, None)
#         while current_phoneme:
#             phoneme_end_time = current_phoneme.end_time
#             if phoneme_end_time == word_end_time:
#                 last_phoneme_flags.append((current_phoneme.text, 1))
#                 break
#             else:
#                 last_phoneme_flags.append((current_phoneme.text, 0))
#             current_phoneme = next(phoneme_iter, None)
#     return last_phoneme_flags
