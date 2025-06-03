import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra

# === Matcha ===
from matcha.models.matcha_tts import MatchaTTS
from matcha.hifigan.models import Generator as MatchaVocoder
from matcha.hifigan.config import v1 as matcha_hifigan_config
from matcha.hifigan.denoiser import Denoiser as MatchaDenoiser
from matcha.text import text_to_sequence as matcha_text_to_sequence
from matcha.utils.utils import intersperse

# === Tacotron2 ===
from speechbrain.inference.TTS import Tacotron2 as SBTacotron2
from speechbrain.inference.vocoders import HIFIGAN as SBHifiGAN
from text import text_to_sequence as tacotron_text_to_sequence

# === BigVGAN ===
from bigvgan import BigVGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    print("Using config:\n", OmegaConf.to_yaml(cfg))
    text = cfg.inference.text

    # === Load model ===
    if cfg.model.type == "matcha":
        model = MatchaTTS.load_from_checkpoint(cfg.model.checkpoint, map_location=device).eval().to(device)
        seq = torch.tensor(intersperse(matcha_text_to_sequence(text, ['basic_cleaners'])[0], 0), dtype=torch.long, device=device).unsqueeze(0)
        mel = model.synthesise(seq, torch.tensor([seq.shape[-1]], device=device))[0]['mel']

    elif cfg.model.type == "tacotron2":
        model = SBTacotron2.from_hparams(source=cfg.model.source, savedir=cfg.model.savedir, run_opts={"device": device})
        seq = model.encode_text(text)
        mel = model(seq).squeeze(0).permute(1, 0)  # [T, 80] → [80, T]

    elif cfg.model.type == "ringformer":
        raise NotImplementedError("Ringformer integration must be defined specifically.")
    
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # === Load vocoder ===
    if cfg.vocoder.type == "hifigan":
        if cfg.model.type == "matcha":
            voc = MatchaVocoder(matcha_hifigan_config).to(device)
            voc.load_state_dict(torch.load(cfg.vocoder.checkpoint, map_location=device)['generator'])
            voc.eval().remove_weight_norm()
            denoiser = MatchaDenoiser(voc)
            audio = voc(mel.unsqueeze(0)).squeeze(0)
            audio = denoiser(audio, strength=cfg.vocoder.denoise_strength).squeeze()
        else:
            voc = SBHifiGAN.from_hparams(source=cfg.vocoder.source, savedir=cfg.vocoder.savedir, run_opts={"device": device})
            audio = voc.decode_batch(mel.unsqueeze(0)).squeeze()

    elif cfg.vocoder.type == "bigvgan":
        voc = BigVGAN.from_pretrained(cfg.vocoder.source, use_cuda_kernel=False).to(device).eval()
        voc.remove_weight_norm()
        audio = voc(mel.unsqueeze(0)).squeeze()

    elif cfg.vocoder.type == "griffinlim":
        mel = mel.cpu().detach().numpy()
        mel = np.exp(mel)
        stft = librosa.feature.inverse.mel_to_stft(mel, sr=cfg.vocoder.sampling_rate, n_fft=1024)
        audio = librosa.griffinlim(stft, n_iter=60)

    else:
        raise ValueError(f"Unknown vocoder type: {cfg.vocoder.type}")

    # === Save output ===
    output_path = Path(cfg.inference.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio.cpu().numpy(), cfg.vocoder.sampling_rate)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
