import os
import json
from pathlib import Path


def prepare_from_text_files(train_file: str, valid_file: str, save_folder: str) -> None:
    """
    Parses train and validation transcript files and converts them into JSON format for TTS processing.

    Args:
        train_file (str): Path to the train text file.
        valid_file (str): Path to the validation text file.
        save_folder (str): Folder to save the resulting JSON files.
    """

    def parse_txt(file_path: str, wav_base_path: str = None) -> dict:
        """
        Parses a transcript file into a dictionary suitable for JSON serialization.

        Args:
            file_path (str): Path to the text file.
            wav_base_path (str, optional): Base path to prepend to .wav filenames.

        Returns:
            dict: Dictionary with utterance metadata.
        """
        data = {}
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                wav_rel_path, label = line.strip().split("|", 1)
                uttid = Path(wav_rel_path).stem
                wav_full_path = os.path.join(wav_base_path, wav_rel_path + ".wav") if wav_base_path else wav_rel_path + ".wav"
                data[uttid] = {
                    "uttid": uttid,
                    "wav": wav_full_path,
                    "label": label.strip()
                }
        return data

    # Create output directory
    os.makedirs(save_folder, exist_ok=True)

    # Define base paths for the audio files
    train_wav_base = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS/Text-To-speech/data/saspeech_automatic_data/"
    valid_wav_base = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS/Text-To-speech/data/saspeech_gold_standard/"

    # Parse the transcript text files
    train_data = parse_txt(train_file, wav_base_path=train_wav_base)
    valid_data = parse_txt(valid_file, wav_base_path=valid_wav_base)
    test_data = {}  # Optional placeholder

    # Save to JSON
    with open(os.path.join(save_folder, "train.json"), "w", encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(save_folder, "valid.json"), "w", encoding='utf-8') as f:
        json.dump(valid_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(save_folder, "test.json"), "w", encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Prepared JSONs: {len(train_data)} train, {len(valid_data)} valid.")
