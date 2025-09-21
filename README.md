# CIFAR-10 Image Classification Project

A deep learning project for image classification using the CIFAR-10 dataset, featuring both a Jupyter notebook for training and experimentation, and a web application for interactive image classification.

## Overview

This project implements a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The project includes:

1. **Model Training Notebook**: Complete workflow for data exploration, model building, training, and evaluation
2. **Web Application**: Interactive Gradio interface for real-time image classification
3. **Pre-trained Model**: Saved model weights ready for inference

## Live App Link : https://huggingface.co/spaces/Thiyanesh07/Image_classification

## Project Structure

```
DL_Task/
├── img_classification.ipynb    # Main training notebook
├── app.py                      # Gradio web application
├── img_classification.keras    # Pre-trained model file
├── requirements.txt           # Python dependencies (empty - needs to be populated)
├── LICENSE                    # MIT License
└── README.md                 # Project documentation
```

## Features

### Model Architecture
- **Deep CNN** with batch normalization and dropout for regularization
- **Three convolutional blocks** (32, 64, 128 filters)
- **Global Average Pooling** to reduce parameters
- **Dense layers** with dropout for final classification
- **Total Parameters**: ~324K (1.24 MB)

### Training Details
- **Dataset**: CIFAR-10 (50,000 training + 10,000 test images)
- **Data Augmentation**: Rotation, shifts, horizontal flips, and zoom
- **Optimization**: Adam optimizer with learning rate scheduling
- **Callbacks**: Model checkpointing, early stopping, and learning rate reduction
- **Final Test Accuracy**: 77.24%

### Web Interface
- **Framework**: Gradio for easy-to-use web UI
- **Input**: Upload any image (automatically resized to 32x32)
- **Output**: Probability scores for all 10 CIFAR-10 classes
- **Real-time**: Instant predictions with confidence scores

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Thiyanesh07/DL_Task.git
cd DL_Task
```

2. Install dependencies (requirements.txt needs to be populated):
```bash
# Required packages (add to requirements.txt):
pip install tensorflow gradio pillow numpy matplotlib seaborn scikit-learn
```

## Usage

### Training the Model

1. Open `img_classification.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells to:
   - Load and explore the CIFAR-10 dataset
   - Build and compile the CNN model
   - Train with data augmentation and callbacks
   - Evaluate performance and visualize results
   - Save the trained model

### Running the Web Application

```bash
python app.py
```

The Gradio interface will launch, allowing you to:
- Upload images for classification
- View probability scores for all classes
- Get instant predictions

## Model Performance

- **Test Accuracy**: 77.24%
- **Test Loss**: 0.7085
- **Training Features**: 
  - Data augmentation for better generalization
  - Batch normalization for stable training
  - Dropout for regularization
  - Learning rate scheduling for optimal convergence

## Dataset Information

**CIFAR-10 Classes:**
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

Each class contains 6,000 images (5,000 training + 1,000 test), with images sized 32×32 pixels in RGB format.

## Technical Details

### Model Architecture
```
Conv2D(32) → BatchNorm → ReLU → Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout
Conv2D(64) → BatchNorm → ReLU → Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout
Conv2D(128) → BatchNorm → ReLU → Conv2D(128) → BatchNorm → ReLU → GlobalAvgPool
Dropout → Dense(256) → ReLU → Dropout → Dense(10) → Softmax
```

### Training Configuration
- **Epochs**: 50 (with early stopping)
- **Batch Size**: 64
- **Learning Rate**: 1e-3 (with ReduceLROnPlateau)
- **Validation Split**: 10% of training data
- **Seed**: 42 (for reproducibility)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CIFAR-10 dataset by Alex Krizhevsky
- TensorFlow/Keras for deep learning framework
- Gradio for the web interface
- Google Colab for training environment
