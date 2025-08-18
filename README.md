## Preview
[![Watch the demo](https://github.com/Thaman-N/TDISS/raw/main/frontend/public/demo-video.gif)](https://github.com/Thaman-N/TDISS/raw/main/frontend/public/demo-video.mp4)

# Aggression Detection Dashboard

A web-based dashboard for analyzing videos and detecting aggression using deep learning models. Built with FastAPI and PyTorch, featuring a modern UI with real-time processing and detailed results visualization.

## Features

- **Real-time Processing**: Upload videos or provide file paths for analysis
- **Live Job Tracking**: Monitor processing progress with real-time updates
- **Detailed Results**: View violence timeline, confidence scores, and video metadata
- **Search & Filter**: Browse history with search and filter capabilities
- **Modern UI**: Clean, responsive design using Tailwind CSS

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

# 2. Activate the new environment
conda activate violence_detect

# 3. Go to frontend directory
cd frontend

# 4. Install dependencies
npm install
```

3. **Add your PyTorch model**
```bash
# Use the model in the repo or change the path to your own model in main.py & torch_detection.py
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

6. **Open your browser**
Navigate to http://localhost:5173

## Using the Training Pipeline
```bash
cd backend/trainingpipeline
# Depending on hardware and location of dataset, command and paths in files need to be changed accordingly
# may have to run export KMP_DUPLICATE_LIB_OK="TRUE" or $env:KMP_DUPLICATE_LIB_OK = "TRUE" on powershell
python train_x3d_violence.py --dataset_path "C:\archive\RWF-2000" --batch_size 8 --num_epochs 30 --learning_rate 5e-5 --gradient_clip_val 1.0 --warmup_epochs 3 --scheduler plateau --mixed_precision --checkpoint_dir stable_checkpoints
```

## Project Structure

```
backend
├── main.py
├── model.py
├── torch_detection.py
├── stable_best_model.pth
├── trainingpipeline
│   ├── constraints.txt
│   ├── train_x3d_violence.py
│   ├── x3d_dataset.py
│   ├── x3d_model.py
│   ├── x3d_trainer.py
│   ├── checkpoints
│   │   ├── best_model.pth
│   │   ├── training_curves.png
│   │   └── training_history.json

frontend
├── public
│   ├── 0Ow4cotKOuw_2.avi
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
│   │       ├── button.jsx
│   │       ├── card.jsx
│   │       ├── dialog.jsx
│   │       ├── dropdown-menu.jsx
│   │       ├── input.jsx
│   │       ├── label.jsx
│   │       ├── navigation-menu.jsx
│   │       ├── progress.jsx
│   │       ├── sonner.jsx
│   │       ├── table.jsx
│   │       └── tabs.jsx
│   ├── contexts
│   │   └── WebSocketContext.jsx
│   ├── hooks
│   │   ├── useDarkMode.js
│   │   └── useTheme.js
│   └── lib
│       └── utils.js
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

## Requirements

- NodeJS
- Python 3.11+
- PyTorch
- OpenCV
- FastAPI
- GPU recommended (may need to tweak if you intend to use CPU)



**Note**: This project requires you to train the aggression detection model or use a pretrained one like the file in the repo. Make sure your `model.py` file matches the architecture of the model that was trained.
