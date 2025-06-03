import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from Data_Creation_scripts.HebrewToEnglish import HebrewToEnglish
# === BigVGAN ===
from BigVGAN.bigvgan import BigVGAN

# === Matcha ===
from Matcha_TTS.matcha.models.matcha_tts import MatchaTTS
from Matcha_TTS.matcha.hifigan.models import Generator as MatchaVocoder
from Matcha_TTS.matcha.hifigan.config import v1 as matcha_hifigan_config
from Matcha_TTS.matcha.hifigan.denoiser import Denoiser as MatchaDenoiser
from Matcha_TTS.matcha.text import text_to_sequence as matcha_text_to_sequence
from Matcha_TTS.matcha.utils.utils import intersperse

# === Tacotron2 ===
from speechbrain.inference.TTS import Tacotron2 as Tacotron2
from speechbrain.inference.vocoders import HIFIGAN as SBHifiGAN
from tracatron2.text import text_to_sequence as tacotron_text_to_sequence


# === MatchaTTS model inference ===
@torch.inference_mode()
def infer_matcha(model_cfg, text, device):
    model = MatchaTTS.load_from_checkpoint(model_cfg.checkpoint, map_location=device).eval().to(device)
    seq = torch.tensor(intersperse(matcha_text_to_sequence(text, ['basic_cleaners'])[0], 0),
                       dtype=torch.long, device=device)[None]
    mel = model.synthesise(seq, torch.tensor([seq.shape[-1]], device=device))[0]['mel']
    return mel

# === Tacotron2 model inference ===
@torch.inference_mode()
def infer_tacotron2(model_cfg, text, device):
    model = Tacotron2.from_hparams(source=model_cfg.source, savedir=model_cfg.savedir,
                                   run_opts={"device": device})
    seq = model.encode_text(text)
    mel = model(seq).squeeze(0).permute(1, 0)  # [T, 80] → [80, T]
    return mel

# === Vocoder inference ===
@torch.inference_mode()
def vocode_hifigan(vocoder_cfg, mel, model_type, device):
    if model_type == "matcha":
        voc = MatchaVocoder(matcha_hifigan_config).to(device)
        voc.load_state_dict(torch.load(vocoder_cfg.checkpoint, map_location=device)['generator'])
        voc.eval().remove_weight_norm()
        denoiser = MatchaDenoiser(voc)
        audio = voc(mel.unsqueeze(0)).squeeze(0)
        return denoiser(audio, strength=vocoder_cfg.denoise_strength).squeeze()
    else:
        voc = SBHifiGAN.from_hparams(source=vocoder_cfg.source, savedir=vocoder_cfg.savedir,
                                     run_opts={"device": device})
        return voc.decode_batch(mel.unsqueeze(0)).squeeze()

@torch.inference_mode()
def vocode_bigvgan(vocoder_cfg, mel, device):
    voc = BigVGAN.from_pretrained(vocoder_cfg.source, use_cuda_kernel=False).to(device).eval()
    voc.remove_weight_norm()
    return voc(mel.unsqueeze(0)).squeeze()

@torch.inference_mode()
def vocode_griffinlim(vocoder_cfg, mel):
    mel = mel.cpu().detach().numpy()
    mel = np.exp(mel)
    stft = librosa.feature.inverse.mel_to_stft(mel, sr=vocoder_cfg.sampling_rate, n_fft=1024)
    return librosa.griffinlim(stft, n_iter=60)

# === Main  ===
@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using config:\n", OmegaConf.to_yaml(cfg))
    text = cfg.inference.text
    
    text = HebrewToEnglish(text)  # Convert Hebrew text to English sounds

    # Model inference
    if cfg.model.type == "matcha":
        mel = infer_matcha(cfg.model, text, device)
    elif cfg.model.type == "tacotron2":
        mel = infer_tacotron2(cfg.model, text, device)
    elif cfg.model.type == "ringformer":
        raise NotImplementedError("Ringformer integration must be defined specifically.")
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # Vocoder
    if cfg.vocoder.type == "hifigan":
        audio = vocode_hifigan(cfg.vocoder, mel, cfg.model.type, device)
    elif cfg.vocoder.type == "bigvgan":
        audio = vocode_bigvgan(cfg.vocoder, mel, device)
    elif cfg.vocoder.type == "griffinlim":
        audio = vocode_griffinlim(cfg.vocoder, mel)
    else:
        raise ValueError(f"Unknown vocoder type: {cfg.vocoder.type}")

    # Save audio
    output_path = Path(cfg.inference.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio.cpu().numpy(), cfg.vocoder.sampling_rate)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()