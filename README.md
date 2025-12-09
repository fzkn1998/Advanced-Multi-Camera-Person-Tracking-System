# 🎯 Advanced Multi-Camera Person Tracking System

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, real-time person tracking system that leverages state-of-the-art YOLOv8 for precise detection and DeepSORT for robust multi-object tracking across multiple camera feeds. Ideal for surveillance, retail analytics, and security applications.

## 🌟 Key Features

- **High-Accuracy Detection**: Utilizes YOLOv8 for real-time person detection with excellent precision
- **Multi-Camera Support**: Seamlessly track individuals across multiple camera angles
- **Persistent Tracking**: DeepSORT algorithm ensures consistent ID assignment across frames
- **Real-time Performance**: Optimized for smooth operation on standard hardware
- **Customizable Parameters**: Fine-tune tracking sensitivity and performance
- **Cross-Camera Re-identification**: Track the same person across different camera views

## 🚀 Applications

- **Retail Analytics**: Monitor customer movement and behavior patterns
- **Security & Surveillance**: Track persons of interest across multiple cameras
- **Crowd Management**: Analyze foot traffic and density in public spaces
- **Smart Cities**: Monitor pedestrian flow in urban environments
- **Access Control**: Track individuals in restricted areas

## 🛠️ Technical Stack

- **Detection**: YOLOv8 (State-of-the-art object detection)
- **Tracking**: DeepSORT (Deep Learning based tracking)
- **Backend**: PyTorch
- **Computer Vision**: OpenCV
- **Language**: Python 3.7+

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- CUDA-compatible GPU (recommended for best performance)

### Setup Instructions

1. **Clone the repository**:
   ```bash
   
   git clone https://github.com/fzkn1998/Advanced-Multi-Camera-Person-Tracking-System.git
   cd Advanced-Multi-Camera-Person-Tracking-System
   ```
2. **Create and activate a virtual environment** (recommended):
   ```bash
   
   firstly download miniconda from https://www.anaconda.com/download/success then,
   conda create -n myenv python=3.10
   conda activate myenv
   ```
3. **Install dependencies**:
   ```bash
   
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

1. **Prepare your video files**:
   - Place your video files in the project directory
   - Supported formats: MP4, AVI, or any format supported by OpenCV

2. **Configure the script**:
   Open `test6.py` and update the following parameters:
   ```python
   SRC1 = "input1.mp4"  # Path to your first video file
   SRC2 = "input2.mp4"  # Path to your second video file
   YOLO_CONF = 0.35     # Detection confidence threshold (0-1)
   MIN_BOX_AREA = 900   # Minimum bounding box area to track
   ```

3. **Run the application**:
   ```bash
   python test6.py
   ```

4. **Using the application**:
   - The system will display two windows showing the camera feeds
   - Each detected person will be assigned a unique ID
   - Press 'q' to quit the application

## ⚙️ Advanced Configuration

### Key Parameters
- `YOLO_CONF` (default: 0.35): Higher values increase detection confidence but may miss some objects
- `SIMILARITY_THRESHOLD` (default: 0.99): Adjust for re-identification sensitivity
- `MIN_BOX_AREA` (default: 900): Filter out small detections to reduce noise

### Performance Tuning
- For better performance on CPU, you may need to reduce the input resolution
- On GPU, you can increase the frame processing rate by adjusting the frame skip parameter

## 📊 Expected Results

- Real-time tracking at 15-30 FPS on a modern CPU/GPU
- High accuracy in person detection (>90% mAP)
- Consistent ID assignment across frames
- Minimal ID switches during tracking

## 🏗️ Project Structure

```
.
├── test6.py           # Main application script
├── requirements.txt   # Python dependencies
├── README.md          # This documentation
├── input1.mp4         # Sample video 1 (example)
└── input2.mp4         # Sample video 2 (example)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) for the powerful detection model
- [DeepSORT](https://github.com/nwojke/deep_sort) for the tracking algorithm
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [OpenCV](https://opencv.org/) for computer vision operations

## 📞 Support

For issues, questions, or contributions:

Open an issue on GitHub
Submit a pull request
Contact: fzkn1998@gmail.com


