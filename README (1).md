# Ubuntu Corpus

> **The world-class automated pipeline for African language AI training data.**
> Competing directly with Mozilla Common Voice, Google FLEURS, and Meta VoxPopuli.

[![License](https://img.shields.io/badge/license-Apache%202.0-gold.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-ubuntu--corpus-yellow)](https://huggingface.co/ubuntu-corpus)
[![Quality](https://img.shields.io/badge/quality-world--class-brightgreen.svg)]()
[![Models](https://img.shields.io/badge/models-distil--whisper%2Fdistil--large--v3%20%2B%20MMS--300M-orange.svg)]()

---

## Why Ubuntu Corpus Produces the Best African Language Data in the World

Every other African language dataset has one of these problems:

| Dataset | Problem |
|---------|---------|
| Mozilla Common Voice | Requires 5 human votes per clip → slow, expensive, biased |
| Google FLEURS | Read speech only → unnatural, no dialectal variation |
| Meta VoxPopuli | Parliamentary French/English → 0 African language coverage |
| Random scraped datasets | No quality filtering → hallucinations, wrong language, noise |

**Ubuntu Corpus solves all of this with one automated pipeline.**

---

## The Quality Guarantee

Every single record published has passed a **10-point automated quality gauntlet**:

```
F01 · Duration ≥ 20s
F02 · Word count ≥ 80 words
F03 · Quality score ≥ 0.72/1.00
F04 · Unicode normalization (NFC, invisible chars removed)
F05 · Noise ratio < 20% (low-confidence segments filtered)
F06 · Repetition ratio < 3% (looping audio, jingles rejected)
F07 · Lexical diversity ≥ 0.50 TTR (no repetitive content)
F08 · Language purity < 10% foreign tokens (code-switching filter)
F09 · Audio SNR ≥ 8dB (background noise check)
F10 · Dual-model cross-validation (distil-whisper/distil-large-v3 ↔ MMS-300M)
```

Filter #10 is the key innovation: **two independent AI models must agree** on the transcription before a record gets a Platinum or Gold tier. This is the same dual-signal methodology used internally at Google for FLEURS.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    UBUNTU CORPUS V1 PIPELINE                         │
│                                                                       │
│  Stage 0 · Pre-flight     Environment & dependency validation        │
│  Stage 1 · Harvest        Parallel download (yt-dlp, RSS, direct)   │
│  Stage 2 · Transcribe     distil-whisper/distil-large-v3 (primary)        │
│                           MMS-300M (secondary, cross-validation)     │
│  Stage 3 · Quality        10-point gauntlet → tier assignment       │
│  Stage 4 · Enrich         Dialect detection · speaker estimation     │
│  Stage 5 · Export         JSONL + Parquet + per-tier + per-dialect  │
│  Stage 6 · Publish        HuggingFace Hub (auto dataset card)       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quality Tiers

| Tier | Criteria |
|------|----------|
| 💎 **Platinum** | Quality ≥ 0.96 AND Whisper↔MMS agreement ≥ 88% |
| 🥇 **Gold** | Quality ≥ 0.88 AND agreement ≥ 72% |
| 🥈 **Silver** | Quality ≥ 0.72 AND agreement ≥ 55% |
| ✗ Rejected | Failed any of the 10 filters |

---

## Supported Languages (V1)

| Language | ISO | Region | Speakers | Dialects |
|----------|-----|--------|----------|----------|
| Swahili | `sw` | East Africa | 200M+ | sw-TZ, sw-KE, sw-CD, sw-UG |
| Hausa | `ha` | West Africa | 80M+ | ha-NG, ha-NE, ha-GH |
| Fula | `ff` | West Africa | 40M+ | ff-SN, ff-GN, ff-ML, ff-NG |
| Lingala | `ln` | Central Africa | 40M+ | ln-CD, ln-CG, ln-CF |
| Yoruba | `yo` | West Africa | 50M+ | yo-NG, yo-BJ, yo-GH |
| Igbo | `ig` | West Africa | 35M+ | ig-NG |
| Amharic | `am` | Horn of Africa | 30M+ | am-ET |
| Bambara | `bm` | West Africa | 15M+ | bm-ML, bm-GN |
| Wolof | `wo` | West Africa | 12M+ | wo-SN, wo-GM |
| Twi | `ak` | West Africa | 10M+ | ak-GH |
| Fon | `fon` | West Africa | 5M+ | fon-BJ |

**Adding a new language = 6 lines in `config.yaml`.**

---

## Quick Start

### 1. Install system dependencies

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### 2. Install Python dependencies

```bash
# Clone
git clone https://github.com/ubuntu-corpus/pipeline
cd pipeline

# Full world-class stack (recommended)
pip install -r requirements.txt

# Minimal (no cross-validation, no language filtering)
pip install openai-whisper yt-dlp pyyaml
```

### 3. Run

```bash
# World-class run (best quality — requires ~600MB RAM for distil-large-v3):
python pipeline.py --model distil-whisper/distil-large-v3 --use-mms

# Publish to HuggingFace:
python pipeline.py --model distil-whisper/distil-large-v3 --use-mms --hf-token hf_yourtoken

# GPU accelerated (10x faster):
python pipeline.py --model distil-whisper/distil-large-v3 --device cuda --use-mms

# Fast test run (lower quality, faster):
python pipeline.py --model base --no-mms

# Skip harvesting, re-run quality gauntlet on cached transcripts:
python pipeline.py --skip-harvest --skip-transcribe
```

---

## Dataset Format

Each record in `data.jsonl` — richer than Mozilla Common Voice:

```json
{
  "id":                    "uc_a1b2c3d4e5f6g7",
  "language":              "Hausa",
  "language_code":         "ha",
  "dialect":               "ha-NG",
  "region":                "West Africa",
  "country":               "NG",
  "domain":                "news",
  "text":                  "Labaran yau sun faro da...",
  "normalized_text":       "Labaran yau sun faro da...",
  "word_count":            847,
  "char_count":            4231,
  "duration_seconds":      312.5,
  "quality_score":         0.9418,
  "tier":                  "gold",
  "crossval_agreement":    0.84,
  "crossval_tier":         "gold",
  "lexical_diversity":     0.6712,
  "noise_ratio":           0.031,
  "repeat_ratio":          0.008,
  "snr_db":               22.4,
  "silence_ratio":         0.12,
  "lid_is_pure":           true,
  "lid_foreign_ratio":     0.018,
  "speaker_count_estimate": 2,
  "source_url":            "https://youtube.com/...",
  "source_platform":       "youtube",
  "harvested_at":          "2026-04-06T09:14:22+00:00",
  "transcribed_at":        "2026-04-06T09:18:47+00:00",
  "pipeline_version":      "1.0.0",
  "segment_count":         84,
  "segments": [
    {
      "id":              "uc_a1b2c3_seg00001",
      "start":           0.0,
      "end":             4.2,
      "text":            "Labaran yau sun faro da...",
      "normalized_text": "Labaran yau sun faro da...",
      "language":        "ha",
      "confidence":      0.9411,
      "is_clean":        true
    }
  ]
}
```

### Fields compared to competitors

| Field | Ubuntu Corpus | Mozilla CV | FLEURS |
|-------|:---:|:---:|:---:|
| Text | ✅ | ✅ | ✅ |
| Time-aligned segments | ✅ | ❌ | ✅ |
| Quality score | ✅ | ✅ | ✅ |
| Quality tier | ✅ | ❌ | ❌ |
| Dual-model agreement | ✅ | ❌ | ❌ |
| Lexical diversity | ✅ | ❌ | ❌ |
| Dialect tag | ✅ | ❌ | ✅ |
| SNR dB | ✅ | ❌ | ❌ |
| Language purity score | ✅ | ❌ | ❌ |
| Speaker count estimate | ✅ | ❌ | ❌ |
| Parquet export | ✅ | ❌ | ✅ |

---

## Output Files

```
data/
├── audio/
│   └── uc_[id].mp3               Downloaded audio files
└── datasets/
    └── hausa/
        ├── data.jsonl            Full dataset (all tiers)
        ├── data_platinum.jsonl   Tier Platinum only
        ├── data_gold.jsonl       Tier Gold only
        ├── data_silver.jsonl     Tier Silver only
        ├── dialect_ha_ng.jsonl   Nigeria dialect split
        ├── dialect_ha_ne.jsonl   Niger dialect split
        ├── train.jsonl           80% split (same as FLEURS)
        ├── validation.jsonl      10% split
        ├── test.jsonl            10% split
        ├── data.parquet          Columnar (BigQuery/Databricks)
        └── README.md             Rich HuggingFace dataset card
```

---

## Adding a New Language

Edit `config.yaml`:

```yaml
languages:
  - name: "Zulu"
    code: "zu"
    region: "Southern Africa"
    sources:
      - url: "https://www.youtube.com/@SABCZulu"
        domain: "news"
        country: "ZA"
```

Run `python pipeline.py` — done. The pipeline automatically:
- Downloads audio
- Transcribes with Whisper + MMS
- Runs all 10 quality filters
- Detects dialect (zu-ZA)
- Exports to JSONL + Parquet + train/val/test splits
- Publishes to HuggingFace with a full dataset card

---

## Using Published Datasets

```python
from datasets import load_dataset

# Standard load
ds = load_dataset("ubuntu-corpus/hausa")
print(ds["train"][0]["text"])

# Load only Platinum tier (highest quality)
ds_platinum = load_dataset(
    "ubuntu-corpus/hausa",
    data_files={"train": "data_platinum.jsonl"}
)

# Filter by quality score
high_quality = ds["train"].filter(lambda x: x["quality_score"] >= 0.90)

# Filter by dialect
nigeria_only = ds["train"].filter(lambda x: x["dialect"] == "ha-NG")

# Use for ASR training (Wav2Vec2, Whisper fine-tuning)
for record in ds["train"]:
    audio_path = record["source_url"]  # or use local audio_path
    text = record["normalized_text"]   # use normalized_text for training
    segments = record["segments"]       # time-aligned for forced alignment
```

---

## Project Structure

```
ubuntu-corpus/
├── pipeline.py        Main pipeline (all 6 stages)
├── config.yaml        Languages, sources, quality thresholds
├── requirements.txt   Python dependencies
├── index.html         Project landing page
└── README.md          This file
```

---

## Roadmap

| Phase | Timeline | Goal |
|-------|----------|------|
| **V1** | **Now** | **11 languages · 10-point QA · Dual-model · World-class** |
| V2 | Month 3 | Discovery Engine (auto-finds new sources) · 30 languages |
| V3 | Month 6 | Premium API · 50+ languages · Enterprise sales |
| V4 | Month 12 | African language model suite · Acquisition conversations |

---

## Why "Ubuntu"?

*Ubuntu* is an African philosophical concept: **"I am because we are."**

Every voice added to this corpus makes African AI more intelligent.
Every dataset published is a step toward AI that speaks *to* Africans, not past them.

---

## License

**Apache 2.0** — Free for research and commercial use.

---

## Citation

```bibtex
@software{ubuntu_corpus_2026,
  title   = {Ubuntu Corpus: World-Class Automated African Language AI Training Data},
  author  = {Ubuntu Corpus Project},
  year    = {2026},
  url     = {https://github.com/ubuntu-corpus/pipeline},
  license = {Apache-2.0},
  note    = {V1. 10-point quality gauntlet. distil-whisper/distil-large-v3 + MMS-300M cross-validation.}
}
```

---

*Built by one founder. Competing with the best labs in the world. For 1.4 billion people.*
