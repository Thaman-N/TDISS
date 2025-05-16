# Violence Detection Dashboard

A web-based dashboard for analyzing videos and detecting violence using deep learning models. Built with Flask and PyTorch, featuring a modern UI with real-time processing and detailed results visualization.

## 🚀 Features

- **Real-time Processing**: Upload videos or provide file paths for analysis
- **Live Job Tracking**: Monitor processing progress with real-time updates
- **Detailed Results**: View violence timeline, confidence scores, and video metadata
- **Search & Filter**: Browse history with search and filter capabilities
- **Modern UI**: Clean, responsive design using Tailwind CSS

## 🛠️ Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Thaman-N/TDISS
cd violence-detection-dashboard
```

2. **Install dependencies**
```bash
pip install flask torch torchvision opencv-python numpy pillow scipy
#or use the environment.yml file
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

## 📁 Project Structure

```
violence-dashboard/
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

## 🔧 Configuration

Adjust detection sensitivity in `app.py`:
```python
app.config['DETECTION_THRESHOLD'] = 0.6  # Default: 0.6 (60%)
```

## 📋 Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Flask
- GPU recommended (but not required)


## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

**Note**: This project requires you to train the violence detection model or use a pretrained one like the file in the repo. Make sure your `model.py` file matches the architecture used during training.
