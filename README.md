# TransMNet
# README for MobileNet-Swin Transformer Hybrid Model

## Overview
This project implements a hybrid deep learning model combining MobileNetV2 as a feature extractor and a Swin Transformer for classification. The model is applied to a sugarcane disease and deficiency dataset, utilizing a combination of PyTorch and TensorFlow for data preprocessing and model training.

### Key Features
1. **MobileNetV2**: Used as a feature extractor with a 1x1 convolutional layer to reduce the number of channels.
2. **Swin Transformer**: Processes extracted features using window-based and shifted window attention mechanisms.
3. **Custom Dataset Handling**: Supports loading and preprocessing of sugarcane dataset using TensorFlow's `ImageDataGenerator` and PyTorch's `DataLoader`.
4. **Metrics and Evaluation**: Computes metrics like accuracy, precision, recall, F1 score, sensitivity, Matthews correlation coefficient (MCC), and cross-correlation coefficient.
5. **Visualization**: Plots training/validation losses and accuracies, as well as confusion matrices for model evaluation.

---

## Requirements

### Python Libraries
The following Python libraries are required:
- torch
- torchvision
- tensorflow
- numpy
- sklearn
- matplotlib

### Hardware Requirements
- GPU with CUDA support is recommended for faster training.

---

## Dataset

### Input Data
The dataset should be organized in the following structure for `ImageDataGenerator`:
```
Dataset Directory/
  class_1/
    image1.jpg
    image2.jpg
    ...
  class_2/
    image1.jpg
    image2.jpg
    ...
  class_3/
    image1.jpg
    image2.jpg
    ...
```

### Preprocessing
- Resized to (224, 224) to match the input size of MobileNet.
- Normalized using mean = [0.5, 0.5, 0.5] and std = [0.5, 0.5, 0.5].

---

## Model Architecture

### MobileNetV2 Feature Extractor
- Pretrained MobileNetV2 is used to extract features up to the last feature layer.
- A 1x1 convolutional layer reduces the channel dimensions to 768.

### Swin Transformer
- Applies window-based attention and shifted window attention.
- Includes feed-forward layers with dropout regularization.
- Outputs class probabilities through a fully connected layer.

### Combined Model
- MobileNetV2 provides spatial features, which are fed into the Swin Transformer for further processing and classification.

---

## Training and Evaluation

### Training
- **Loss Function**: CrossEntropyLoss.
- **Optimizer**: Adam with learning rate `1e-6` and weight decay `1e-3`.
- **Batch Size**: 64.
- **Epochs**: 50.

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- Sensitivity (per class and overall)

### Results Visualization
- Loss and accuracy curves for training and validation phases.
- Normalized confusion matrix with percentage values.

---

## Usage Instructions

### Clone the Repository
```
git clone https://github.com/your-username/mobilenet-swin-transformer.git
cd mobilenet-swin-transformer
```

### Install Requirements
```
pip install -r requirements.txt
```

### Prepare Dataset
Ensure your dataset is organized as described above and update the dataset path in the script:
```python
dataset_path = '/path/to/your/dataset'
```

### Run the Training Script
```
python train_model.py
```

### Evaluate the Model
The script automatically evaluates the model after training and displays metrics and plots.

---

## File Structure
```
.
|-- train_model.py        # Main script for training and evaluation
|-- MobileNetSwin.py      # Model definition
|-- requirements.txt      # List of dependencies
|-- README.md             # Project documentation
```

---

## Acknowledgments
- **PyTorch**: For deep learning model implementation.
- **TensorFlow**: For data preprocessing.
- **MobileNetV2**: Pretrained model for feature extraction.
- **Swin Transformer**: Adapted for image classification.

---

## License
This project is licensed under the MIT License.

---

## Author
Developed by Subhasish Sarkar, Gautam Kumar. Feel free to contact us for questions or contributions!

## Citation
If you are using this code/part of the code, please cite this work as
@article{zu2024reswint,
  title={Self-Attention Enhanced Depthwise Separable Convolution for Early Detection of Sugarcane Diseases and Nutrient Deficiencies},
  author={Sarkar, Subhasish and Kumar, Gautam},
  journal={The Visual Computer},
  volume={},
  pages={},
  year={},
  publisher={Springer}
}

