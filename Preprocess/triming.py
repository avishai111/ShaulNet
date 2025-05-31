import os
from typing import Union
from glob import glob
from tqdm import tqdm
import librosa
import soundfile as sf

def trim_silence_in_directory(input_dir: Union[str, os.PathLike], output_dir: Union[str, os.PathLike], top_db: int = 20) -> None:
    os.makedirs(output_dir, exist_ok=True)

    wav_files = glob(os.path.join(input_dir, "*.wav"))
    print(f"נמצאו {len(wav_files)} קבצים בתיקייה '{input_dir}'")

    for wav_path in tqdm(wav_files, desc="חיתוך קבצים"):
        try:
            y, sr = librosa.load(wav_path, sr=22050)
            yt, _ = librosa.effects.trim(y, top_db=top_db)

            file_name = os.path.basename(wav_path)
            out_path = os.path.join(output_dir, file_name)

            sf.write(out_path, yt, sr)
          #  print(f"חיתוך והמרה: {file_name}")
        except Exception as e:
            print(f"שגיאה בקובץ {wav_path}: {e}")

    print(f"\n✅ סיום! קבצים נשמרו בתיקייה: {output_dir}")

# שימוש:
# שים את הנתיב שאתה רוצה כאן:
input_folder = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/data/saspeech_gold_standard/wavs"
output_folder = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/data/trimm/saspeech_gold_standard/wavs"

trim_silence_in_directory(input_folder, output_folder)

input_folder = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/data/saspeech_automatic_data/wavs"
output_folder = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/data/trimm/saspeech_automatic_data/wavs"

trim_silence_in_directory(input_folder, output_folder)