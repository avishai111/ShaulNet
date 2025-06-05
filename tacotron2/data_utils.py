import random
import numpy as np
import torch
import torch.utils.data

try:
    import layers
    from utils import load_wav_to_torch, load_filepaths_and_text
    from text import text_to_sequence
except ImportError:
    from . import layers
    from .utils import load_wav_to_torch, load_filepaths_and_text
    from .text import text_to_sequence


def apply_mel_augmentation(mel: torch.Tensor, hparams) -> torch.Tensor:
    """
    Apply SpecAugment-style frequency and time masking to a mel-spectrogram.
    """
    if torch.rand(1).item() >= hparams.augment_mel_prob:
        return mel  # ללא אוגמנטציה

    mel = mel.clone()
    num_mel_channels, num_frames = mel.shape

    # Frequency Masking
    for _ in range(hparams.augment_mel_num_masks):
        f = np.random.randint(0, hparams.augment_mel_max_freq_width)
        if f == 0:
            continue
        f0 = np.random.randint(0, max(1, num_mel_channels - f))
        mel[f0:f0 + f, :] = 0

    # Time Masking
    for _ in range(hparams.augment_mel_num_masks):
        t = np.random.randint(0, hparams.augment_mel_max_time_width)
        if t == 0:
            continue
        t0 = np.random.randint(0, max(1, num_frames - t))
        mel[:, t0:t0 + t] = 0

    return mel


def apply_audio_augmentation(audio: torch.Tensor, sampling_rate: int, hparams) -> torch.Tensor:
    """
    Apply waveform-level augmentations to raw audio using hparams.
    """
    audio = audio.clone()

    if audio.dim() > 1:
        audio = audio.squeeze(0)

    if torch.rand(1).item() < hparams.augment_noise_prob:
        noise = torch.randn_like(audio) * hparams.augment_noise_level
        audio += noise

    if torch.rand(1).item() < hparams.augment_gain_prob:
        gain = np.random.uniform(hparams.augment_gain_min, hparams.augment_gain_max)
        audio *= gain

    if torch.rand(1).item() < hparams.augment_speed_prob:
        try:
            import torchaudio
            speed = np.random.uniform(hparams.augment_speed_min, hparams.augment_speed_max)
            new_sr = int(sampling_rate * speed)
            audio = torchaudio.functional.resample(audio, orig_freq=sampling_rate, new_freq=new_sr)

            if new_sr != sampling_rate:
                audio = torchaudio.functional.resample(audio, orig_freq=new_sr, new_freq=sampling_rate)
        except Exception as e:
            print(f"⚠️ Resample failed: {e}")

    return torch.clamp(audio, -1.0, 1.0)







class TextMelLoader(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams,use_augmentation_mel=False, use_augmentation_audio=False):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

        self.use_augmentation_audio = use_augmentation_audio
        self.use_augmentation_mel = use_augmentation_mel
        self.hparams = hparams
        
        if self.use_augmentation_mel or self.use_augmentation_audio:
                print("Using data augmentation audio:", self.use_augmentation_audio)
                print("Using data augmentation mel:", self.use_augmentation_mel)
                print("Using load_mel_from_disk:", self.load_mel_from_disk)
        
        
    def get_mel_text_weight_pair(self, audiopath_and_text):
        if len(audiopath_and_text) == 3:
            audiopath, text, weight = audiopath_and_text
            weight = float(weight)
        else:
            audiopath, text = audiopath_and_text
            weight = 1.0  # ברירת מחדל אם אין עמודת משקל

        text = self.get_text(text)
        mel = self.get_mel(audiopath)

        return (text, mel, torch.tensor(weight, dtype=torch.float32))

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError(f"{sampling_rate} SR doesn't match target {self.stft.sampling_rate}")
            audio_norm = audio / self.max_wav_value
            if self.use_augmentation_audio and torch.rand(1).item() < self.hparams.augment_audio_prob:
                audio_norm = apply_audio_augmentation(audio_norm, self.sampling_rate, self.hparams)
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename, allow_pickle=True))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                f'Mel dimension mismatch: given {melspec.size(0)}, expected {self.stft.n_mel_channels}'
            )
            if self.use_augmentation_mel and torch.rand(1).item() < self.hparams.augment_mel_prob:
                melspec = apply_mel_augmentation(melspec, self.hparams)
        return melspec

    def get_text(self, text):
        return torch.IntTensor(text_to_sequence(text, self.text_cleaners))

    def __getitem__(self, index):
        return self.get_mel_text_weight_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)



class TextMelCollate():
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        # batch: [text, mel, weight]
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        weights = torch.FloatTensor(len(batch))  # נוסיף את המשקלים

        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            weight = batch[ids_sorted_decreasing[i]][2]
            text_padded[i, :text.size(0)] = text
            weights[i] = weight

        # Mel
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, weights
