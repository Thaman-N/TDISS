# Aggression Detection Dashboard

<!-- [![GitHub Stars](https://img.shields.io/github/stars/Thaman-N/TDISS?style=social)](https://github.com/Thaman-N/TDISS/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Thaman-N/TDISS?style=social)](https://github.com/Thaman-N/TDISS/network/members) -->
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Node.js](https://img.shields.io/badge/node-%3E=18.0-brightgreen)](https://nodejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal.svg)](https://fastapi.tiangolo.com/)

---

## Single Video Upload

[![Watch the demo](https://github.com/Thaman-N/TDISS/raw/main/frontend/public/demo-svu.gif)](https://github.com/Thaman-N/TDISS/raw/main/frontend/public/demo-svu.mp4)

---

## Multi Video Upload

[![Watch the demo](https://github.com/Thaman-N/TDISS/raw/main/frontend/public/demo-mvu.gif)](https://github.com/Thaman-N/TDISS/raw/main/frontend/public/mvu-video.mp4)

---

## Multiple Live Streams - Testing occlusion, glare & reflections

[![Watch the demo](https://github.com/Thaman-N/TDISS/raw/main/frontend/public/demo-mls.gif)](https://github.com/Thaman-N/TDISS/raw/main/frontend/public/demo-mls.mp4)

---

## Table of Contents

- [Aggression Detection Dashboard](#aggression-detection-dashboard)
  - [Single Video Upload](#single-video-upload)
  - [Multi Video Upload](#multi-video-upload)
  - [Multiple Live Streams - Testing occlusion, glare \& reflections](#multiple-live-streams---testing-occlusion-glare--reflections)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Benchmark](#benchmark)
  - [Quick Start](#quick-start)
  - [Training Pipeline](#training-pipeline)
  - [Running Tests](#running-tests)
  - [Project Structure](#project-structure)
  - [Requirements](#requirements)
  - [License](#license)

---

## Overview

A web-based dashboard for analyzing videos and detecting aggression using deep learning models. Built with **FastAPI** and **PyTorch**, featuring a modern UI with real-time processing and detailed results visualization.

---

## Features

* **Real-time Processing**: Seamlessly process live RTSP streams or uploaded video files.
* **Live Tracking Dashboards**: Monitor live streams or job progress while it happens.
* **Instant Alerts**: Receive real-time notifications on the UI and Discord, complete with video playbacks and evidence thumbnails.
* **Comprehensive Analytic**: Review & download detailed timelines, confidence scores, and rich metadata for every incident.
* **Event Stitching**: Automatic stitching of sequential detections into single, manageable incidents for clearer reporting.
* **Search & Filter**: History browsing with search and filtering.
* **Modern UI**: Responsive TailwindCSS-based design.

---

## Benchmark

- RWF 2000 - 94.25% Validation Accuracy (New SOTA benchmark)
- RLVS - 99.75% Validation Accuracy (New SOTA benchmark)
- Hockey Fight Videos - 100% Validation Accuracy (SOTA Performance)
- ViolentFlows - 100% Validation Accuracy (SOTA Performance)
- Cross Dataset Validation Accuracy varies from 80-90%

---

## Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/Thaman-N/TDISS
cd TDISS
```

2. **Install dependencies**

```bash
# Create conda environment
conda env create -f environment.yaml

# Activate environment
conda activate violence_detect

# Install pip packages
pip install -r requirements.txt

# Install PyTorch separately(Only run 1 of the following)
pip install -c constraints.txt torch torchvision torchaudio #cpu build
pip install -c constraints.txt torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 #cuda build

# If you run into slowapi or fvcore import errors, run - pip install slowapi fvcore

# Frontend setup
cd frontend
npm install
```

3. **Add your PyTorch model**

```bash
# Ignore if you want to use the one in the repo
cp /path/to/your/model.pth .
```

4. **Run the backend server**

```bash
cd backend
python main.py
```

5. **Setup Discord webhook(optional, only if you want notifications pushed)**

   * Step 1: Create Discord Server

     - Open Discord (web, desktop, or mobile)
     - Click the "+" button on the left sidebar
     - Choose "Create My Own" → "For me and my friends"
     - Name it something like "Security Alerts" or whatever you like
     - Click "Create"

   * Step 2: Create a Channel for Alerts

     - In your new server, right-click the text channels area
     - Click "Create Channel"
     - Name it "stream-alerts" or whatever you like
     - Make sure it's a Text Channel
     - Click "Create Channel"

   * Step 3: Set Up Webhook

     - Right-click on your new channel
     - Click "Edit Channel"
     - Go to "Integrations" tab on the left
     - Click "Webhooks"
     - Click "New Webhook"
     - Name it "EzurNet Bot" or whatever name you prefer
     - Copy the "Webhook URL" - this is what you'll need to add in main.py
     - Save Changes

   ```bash
   #check if the webhook works
   curl -X POST "YOUR_WEBHOOK_URL_HERE" \
     -H "Content-Type: application/json" \
     -d '{"content": "Test message from violence detection system"}'
   ```

   * Step 4: Change webhook URL in main.py

6. **Run the frontend server**

```bash
cd frontend
npm run dev
```

7. **Open in browser** → [http://localhost:5173](http://localhost:5173)

---

## Training Pipeline

```bash
# Older GPU architectures (pre-blackwell) may not allow you to use num_workers argument in which case set it to 0 when running the command
cd backend
# RWF 2000
python trainingpipeline/train_x3d_violence.py --dataset_path "C:\archive\RWF-2000" --batch_size 8 --num_epochs 30 --learning_rate 5e-5 --gradient_clip_val 1.0 --warmup_epochs 3 --scheduler plateau --mixed_precision --checkpoint_dir train_checkpoints --num_workers 8 --spatial_size 336

# RLVS
python datasetsplitfiles/split_rlvs.py --rlvs_path "C:\archive\Real Life Violence Dataset" --output_path "C:\archive\RealLifeViolenceDatasetSplit" --copy
python trainingpipeline/train_x3d_violence.py --dataset_path "C:\archive\RealLifeViolenceDatasetSplit" --batch_size 8 --num_epochs 30 --learning_rate 5e-5 --gradient_clip_val 1.0 --warmup_epochs 3 --scheduler plateau --mixed_precision --checkpoint_dir train_checkpoints --num_workers 8 --spatial_size 336

# Hockey Fights
python datasetsplitfiles/hockey_split.py --input "C:\archive\HockeyFight" --output "C:\archive\HockeyFightSplit"
python trainingpipeline/train_x3d_violence.py --dataset_path "C:\archive\HockeyFightSplit" --batch_size 8 --num_epochs 30 --learning_rate 5e-5 --gradient_clip_val 2.0 --warmup_epochs 3 --scheduler plateau --mixed_precision --checkpoint_dir train_checkpoints --num_workers 8 --spatial_size 224

# ViolentFlows
python datasetsplitfiles/violentflows_split.py --input "C:\archive\ViolentFlows" --output "C:\archive\ViolentFlowsSplit"
python trainingpipeline/train_x3d_violence.py --dataset_path "C:\archive\ViolentFlowsSplit" --batch_size 8 --num_epochs 30 --learning_rate 5e-5 --gradient_clip_val 2.0 --warmup_epochs 3 --scheduler plateau --mixed_precision --checkpoint_dir train_checkpoints --num_workers 8 --spatial_size 224

```

---

## Running Tests

**Testing Model Performance on Val split of a dataset**
```bash
# This can be used to get measure variance of a model on a dataset's val split or to test cross dataset accuracy
cd backend/trainingpipeline
python testval.py --dataset "path/to/dataset" --model "path/to/model.pth" --runs 5 --output "my_custom_output_folder"
```

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

## Project Structure

```
backend
├── main.py
├── model.py
├── torch_detection.py
├── datasetsplitfiles
│   ├── hockey_split.py
│   ├── split_rlvs.py
│   └── violentflows_split.py
├── models
│   ├── hfrand100.pth
│   ├── hfrand100e30.pth
│   ├── hfs429750.pth
│   ├── rlvs9950.pth
│   ├── rwf9425.pth
│   ├── vfrand100.pth
│   ├── vfrand100e28.pth
│   └── vfs4298.pth
├── trainingpipeline
│   ├── testval.py
│   ├── train_x3d_violence.py
│   ├── x3d_dataset.py
│   ├── x3d_model.py
│   └── x3d_trainer.py
└── tests
    ├── pytest.ini
    ├── run_tests.py
    ├── test_api.py
    ├── test_database.py
    ├── test_detection.py
    ├── test_model.py
    ├── test_requirements.txt
    └── test_utils.py

frontend
├── public
│   ├── demo-video.mp4
│   └── vite.svg
├── src
│   ├── App.css
│   ├── App.jsx
│   ├── index.css
│   ├── main.jsx
│   ├── components
│   │   ├── LiveStreamDashboard.jsx
│   │   ├── LandingPage.jsx
│   │   ├── Navigation.jsx
│   │   ├── ProcessingDashboard.jsx
│   │   ├── ResultsViewer.jsx
│   │   ├── StreamFullScreen.jsx
│   │   ├── UploadInterface.jsx
│   │   ├── react-bits
│   │   │   ├── Animations
│   │   │   │   └── MetallicPaint
│   │   │   │       └── MetallicPaint.jsx
│   │   │   ├── Backgrounds
│   │   │   │   └── LightRays
│   │   │   │       └── LightRays.jsx
│   │   │   └── Components
│   │   │       └── MagicBento
│   │   │           └── MagicBento.jsx
│   │   └── ui
│   │       ├── alert.jsx
│   │       ├── badge.jsx
│   │       ├── breadcrumb.jsx
│   │       ├── button.jsx
│   │       ├── card.jsx
│   │       ├── chart.jsx
│   │       ├── dialog.jsx
│   │       ├── dropdown-menu.jsx
│   │       ├── input.jsx
│   │       ├── label.jsx
│   │       ├── navigation-menu.jsx
│   │       ├── progress.jsx
│   │       ├── separator.jsx
│   │       ├── sheet.jsx
│   │       ├── skeleton.jsx
│   │       ├── sonner.jsx
│   │       ├── table.jsx
│   │       ├── tabs.jsx
│   │       └── tooltip.jsx
│   ├── contexts
│   │   └── WebSocketContext.jsx
│   ├── hooks
│   │   ├── useDarkMode.js
│   │   └── useTheme.js
│   └── lib
│       └── utils.js
├── tests
│   ├── integration
│   │   └── App.integration.test.jsx
│   ├── unit
│   │   ├── LandingPage.test.jsx
│   │   ├── Navigation.test.jsx
│   │   ├── ProcessingDashboard.test.jsx
│   │   ├── ResultsViewer.test.jsx
│   │   ├── UploadInterface.test.jsx
│   │   ├── utils.test.js
│   │   └── WebSocketContext.test.jsx
│   ├── setup.jsx
│   └── test-utils.jsx
├── .gitignore
├── components.json
├── eslint.config.js
├── index.html
├── jsconfig.json
├── jsrepo.json
├── package-lock.json
├── package.json
├── postcss.config.js
├── README.md
├── tailwind.config.js
└── vite.config.js
```

---

## Requirements

* Node.js ≥ 18
* Python 3.11+
* PyTorch
* OpenCV
* FastAPI
* GPU recommended (if you do not intend to use gpu, training pipeline might need to be modified a bit but inference pipeline will work regardless)

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

* Commercial use is **not permitted** unless modifications are shared under the same license.
* You are free to use, modify, and distribute this software for research and personal purposes, provided that derivative works are also licensed under AGPL-3.0.

See the full license text in the [LICENSE](./LICENSE) file.

---
