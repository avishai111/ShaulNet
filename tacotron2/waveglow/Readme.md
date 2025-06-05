# 🤖 Robo-Shaul-TTS

**Robo-Shaul-TTS** is an open-source Text-to-Speech (TTS) system designed to synthesize natural-sounding speech from raw text. It supports Hebrew and can be extended to other languages. Built on top of modern neural architectures, it is suitable for both research and practical applications.

---

## 🚀 Features

* 🎙️ High-quality text-to-speech synthesis
* 🌐 Full Hebrew language support (including diacritics and grammar-aware processing)
* 🧠 Based on state-of-the-art models such as: Tacotron2 and HiFi-GAN, BigVGAN,RingFormer.
* 🔊 Optional speaker embedding support for voice customization
* 🧪 Designed for research, development, and deployment


## 📝 Basic Usage

```python
from robo_shaul_tts import synthesize

text = "מה קורה, שאול?"
audio_path = synthesize(text, speaker_id="default")
```

---

## 🛠️ Project Structure

```
Robo-Shaul-TTS/
│
├── models/              # Pretrained model checkpoints
├── robo_shaul_tts/      # Core source code
│   ├── inference.py     # Main TTS inference logic
│   ├── preprocessing/   # Text cleaning, normalization, and tokenization
│   └── utils/           # Utility functions
├── samples/             # Example audio outputs
├── notebooks/           # Jupyter notebooks for experimentation
├── requirements.txt     # Python dependencies
└── README.md            # This file
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
📧 [Avishai Weimzman](mailto:Avishai11900@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/avishai-weizman/)
🐙 [GitHub](https://https://github.com/avishai111)