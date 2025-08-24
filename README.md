# Aggression Detection Dashboard

<!-- [![GitHub Stars](https://img.shields.io/github/stars/Thaman-N/TDISS?style=social)](https://github.com/Thaman-N/TDISS/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Thaman-N/TDISS?style=social)](https://github.com/Thaman-N/TDISS/network/members) -->
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Node.js](https://img.shields.io/badge/node-%3E=18.0-brightgreen)](https://nodejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal.svg)](https://fastapi.tiangolo.com/)

---

## Preview

[![Watch the demo](https://github.com/Thaman-N/TDISS/raw/main/frontend/public/demo-video.gif)](https://github.com/Thaman-N/TDISS/raw/main/frontend/public/demo-video.mp4)

---

## Table of Contents

- [Aggression Detection Dashboard](#aggression-detection-dashboard)
  - [Preview](#preview)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Quick Start](#quick-start)
  - [Training Pipeline](#training-pipeline)
  - [Running Tests](#running-tests)
  - [Zero-shot Classification on RLVS](#zero-shot-classification-on-rlvs)
  - [Project Structure](#project-structure)
  - [Requirements](#requirements)
  - [License](#license)

---

## Overview

A web-based dashboard for analyzing videos and detecting aggression using deep learning models. Built with **FastAPI** and **PyTorch**, featuring a modern UI with real-time processing and detailed results visualization.

---

## Features

* **Real-time Processing**: Add RTSP live streams or upload videos.
* **Live Job Tracking**: Monitor progress with real-time updates.
* **Detailed Results**: Violence timeline, confidence scores, metadata.
* **Search & Filter**: History browsing with search and filtering.
* **Modern UI**: Responsive TailwindCSS-based design.

---

## Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/Thaman-N/TDISS
cd TDISS
```

2. **Install dependencies**

```bash
# 1. Create new environment
conda env create -f environment.yaml

# 2. Activate environment
conda activate violence_detect

# 3. Frontend setup
cd frontend
npm install
```

3. **Add your PyTorch model**

```bash
cp /path/to/your/model.pth .
```

4. **Run the backend server**

```bash
cd backend
python main.py
```

5. **Run the frontend server**

```bash
cd frontend
npm run dev
```

6. **Open in browser** → [http://localhost:5173](http://localhost:5173)

---

## Training Pipeline

```bash
cd backend/trainingpipeline
python train_x3d_violence.py --dataset_path "C:\archive\RWF-2000" --batch_size 8 --num_epochs 30 --learning_rate 5e-5 --gradient_clip_val 1.0 --warmup_epochs 3 --scheduler plateau --mixed_precision --checkpoint_dir train_checkpoints --num_workers 8 --spatial_size 336
```

---

## Running Tests

**Backend tests:**

```bash
cd backend/tests
python run_tests.py quick   # Quick tests
python run_tests.py core    # Core tests
python run_tests.py all     # Full suite
python run_tests.py coverage
```

**Frontend tests:**

```bash
cd frontend/tests
npm test                    # Watch mode
npm run test:run            # Run once
npm run test:coverage       # With coverage
```

---

## Zero-shot Classification on RLVS

```bash
python evaluate_rlvs.py "your/dataset/path"
```

---

## Project Structure

```
backend
├── evaluate_rlvs.py
├── main.py
├── model.py
├── torch_detection.py
├── nineone75.pth
├── trainingpipeline
│   ├── testval.py
│   ├── train_x3d_violence.py
│   ├── x3d_dataset.py
│   ├── x3d_model.py
│   └── x3d_trainer.py
└── tests
    ├── run_tests.py
    ├── test_api.py
    ├── test_database.py
    ├── test_detection.py
    ├── test_model.py
    └── test_utils.py

frontend
├── public
│   ├── demo-video.mp4
│   └── vite.svg
├── src
│   ├── App.jsx
│   ├── components
│   ├── contexts
│   ├── hooks
│   └── lib
└── tests
```

---

## Requirements

* Node.js ≥ 18
* Python 3.11+
* PyTorch
* OpenCV
* FastAPI
* GPU recommended for inference

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

* Commercial use is **not permitted** unless modifications are shared under the same license.
* You are free to use, modify, and distribute this software for research and personal purposes, provided that derivative works are also licensed under AGPL-3.0.

See the full license text in the [LICENSE](./LICENSE) file.

---
