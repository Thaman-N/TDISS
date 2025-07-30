# Violence Detection Dashboard

A web-based dashboard for analyzing videos and detecting violence using deep learning models. Built with Flask and PyTorch, featuring a modern UI with real-time processing and detailed results visualization.

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
conda env create -f environment.yml

# 2. Activate the new environment
conda activate violence_detect

# 3. Install PyTorch 2.7.0 with CUDA 12.8 support (needs tweaking if code is to be used without cuda support)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 4. Install open-clip-torch (compatible with PyTorch 2.7)
pip install open-clip-torch==2.32.0

```

3. **Add your PyTorch model**
```bash
# Place your trained model file in the project root or use the one in the repo
cp /path/to/your/model.pth .
```

4. **Run the application**
```bash
python app.py
```

5. **Open your browser**
Navigate to http://localhost:5000

## Project Structure

```
TDISS/
├── app.py              # Flask web application
├── torch_detection.py  # PyTorch model integration
├── templates/          # HTML templates
│   ├── base.html
│   ├── index.html
│   └── result.html
├── uploads/            # Uploaded videos
├── results/            # Analysis results
├── model.py           # Model definition (required)
└── model_final.pth    # Model we trained based on architecture in model.py
```

## Configuration

Adjust detection sensitivity in `app.py`:
```python
app.config['DETECTION_THRESHOLD'] = 0.6  # Default: 0.6 (60%)
```

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Flask
- GPU recommended (but not required)



**Note**: This project requires you to train the violence detection model or use a pretrained one like the file in the repo. Make sure your `model.py` file matches the architecture used during training.
