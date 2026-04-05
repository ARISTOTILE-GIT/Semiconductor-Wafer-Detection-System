# 🔬 Semiconductor Wafer Defect Detection System

A full end-to-end AI-powered wafer defect classification and yield prediction dashboard.

## Features

- **Defect Classification** — EfficientNet-B3 trained on WM-811K dataset (9 defect classes)
- **Murphy Yield Model** — Mathematical yield prediction per wafer
- **LLM Expert Analysis** — Groq LLaMA 3.3 70B explains root cause, severity, corrective action
- **Draw Mode** — Engineer draws defect pattern → instant prediction
- **History Tracking** — All predictions saved with charts and CSV export
- **Yield Analytics** — Yield trend, defect distribution, yield vs die area curve

## Defect Classes

| Class | Description |
|-------|-------------|
| Center | Defects clustered at wafer center |
| Donut | Ring-shaped defect pattern |
| Edge-Loc | Defects at edge locations |
| Edge-Ring | Full edge ring defect |
| Loc | Localized defect cluster |
| Near-full | Nearly full wafer coverage |
| Random | Randomly distributed defects |
| Scratch | Linear scratch defect |
| none | No defect pattern |

## Tech Stack

- **Model**: EfficientNet-B3 (PyTorch) — hosted on HuggingFace
- **Dashboard**: Streamlit
- **NLP**: Groq API (LLaMA 3.3 70B)
- **Yield**: Murphy's Yield Model

## Setup

1. Clone repo
```bash
git clone https://github.com/YOUR_USERNAME/Semiconductor-Wafer-Detection-System
cd Semiconductor-Wafer-Detection-System
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run
```bash
streamlit run app.py
```

4. Add your Groq API key in the sidebar

## Model

Model hosted at: [totz07/wafer-defect-classifier](https://huggingface.co/totz07/wafer-defect-classifier)
