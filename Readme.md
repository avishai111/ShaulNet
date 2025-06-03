# 🤖 Robo-Shaul-TTS
---

**Robo-Shaul-TTS** is an Text-to-Speech (TTS) system designed to synthesize natural-sounding speech from raw text. 
This repository include using : (Matcha-TTS)[!https://github.com/shivammehta25/Matcha-TTS], (Ringformer)[!https://github.com/seongho608/RingFormer/tree/main], (tracatron2)[!https://github.com/NVIDIA/tacotron2], (HiFi-GAN)[!https://github.com/jik876/hifi-gan], (BigVGAN)[!https://github.com/NVIDIA/BigVGAN], (Griffin-lin)[https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.GriffinLim.html], (WaveGlow)[!https://pytorch.org/hub/nvidia_deeplearningexamples_waveglow/]

---

## 🚀 Features

* 🎙️ High-quality text-to-speech synthesis
* 🌐 Full Hebrew language support (including diacritics and grammar-aware processing)
* 🧠 Based on state-of-the-art models such as Tacotron2 and HiFi-GAN
* 🔊 Optional speaker embedding support for voice customization
* 🧪 Designed for research, development, and deployment

---


## 📝 Basic Usage

```bash
# Run TTS using the config file (choose model/vocoder inside it)
python inference.py

---

## 🛠️ Project Structure

```
Robo-Shaul-TTS/
│
├── config/                             # YAML configuration files for Hydra (model/vocoder selection)
├── data/                               # Raw or processed datasets (e.g., text inputs or training data)
├── Data_Creation_scripts/              # Scripts for text preprocessing (e.g., Hebrew-to-English conversion)
├── download_pretrained_models_scripts/ # Scripts to download pretrained models (e.g., tracatron2, waveglow)
├── filelists/                          # File lists for model training (e.g., train.txt, val.txt)
├── Matcha_TTS/                         # Matcha-TTS model codebase (training, synthesis, utils)
├── Plotting/                           # Utility scripts for visualizing mel-spectrograms and alignments
├── Preprocess/                         # Data preprocessing pipeline (cleaners, format converters, etc.)
├── pretrained_models/                  # Folder containing downloaded/pretrained TTS and vocoder models
├── RingFormer/                         # Optional RingFormer model implementation (if enabled)
├── tracatron2/                         # Tacotron2 model code (architecture, training, inference)
├── .gitignore                          # Git ignore rules (e.g., checkpoints, logs, outputs)
├── LICENSE                             # Open-source license (e.g., MIT, Apache)
├── inference.py                        # Main script to perform TTS inference via Hydra-configurable pipeline
└── README.md                           # Main project documentation and usage instructions

```

---

## 🧑‍🔬 Contributing

We welcome contributions of all kinds — code, bug fixes, documentation, and new feature ideas. Feel free to open issues or pull requests!

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 📞 Contact

For questions, issues, or collaboration ideas:
📧 [Avishai Weizman](mailto:Avishai11900@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/avishai-weizman/)
🐙 [GitHub](https://github.com/avishai111)

---

