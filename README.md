# Medical Image Classification: Dermoscopy & Blood Cell Microscopy

A deep learning project using ResNet18 to classify medical images from two different imaging modalities: dermoscopy (skin lesions) and blood cell microscopy.

## Project Overview

This project develops a convolutional neural network (CNN) based on ResNet18 architecture to classify six different medical conditions across two imaging modalities. The model is trained on a balanced dataset using class weights to handle imbalanced data effectively.

## Dataset

The dataset contains **10,629 training samples** across 6 classes:

### Dermoscopy (Skin Lesions)
- **Nevus**: 5,362 images
- **Melanoma**: 890 images  
- **Vascular Lesions**: 116 images

### Blood Cell Microscopy
- **Granulocytes**: 2,305 images
- **Basophils**: 990 images
- **Lymphocytes**: 966 images

**Image Specifications**: 28×28 pixels, RGB (3 channels)

## Project Structure

```
├── Data_2.py                           # Data loading and class distribution analysis
├── Upload_Images.py                    # Convert pixel arrays to PNG images
├── Enhance_Images.py                   # Image upscaling (2x interpolation)
├── Enhanced_Images_to_vectors.py       # Convert enhanced images to NumPy vectors
├── Resnet18.py                         # ResNet18 model definition and training
├── Files/
│   ├── Xtrain_classification2.npy      # Training data (image vectors)
│   ├── Ytrain_classification2.npy      # Training labels
│   ├── Xtest_Classification2.npy       # Test data
│   └── Ytest_Classification2.npy       # Test predictions
└── Images/
    ├── original/                       # Original 28×28 images
    └── enhanced/                       # Upscaled images (2x)
```

## Pipeline

1. **Data Loading** (`Data_2.py`)
   - Load training data from NumPy arrays
   - Analyze class distribution
   - Verify data integrity

2. **Image Conversion** (`Upload_Images.py`)
   - Convert pixel vectors to PNG images
   - Save to disk for visualization

3. **Image Enhancement** (`Enhance_Images.py`)
   - Upscale images using cubic interpolation
   - 2x magnification for improved feature extraction
   - *Note: Enhanced images are not used in final submission due to training time constraints*

4. **Vector Conversion** (`Enhanced_Images_to_vectors.py`)
   - Convert enhanced PNG images back to NumPy vectors
   - Maintain proper ordering based on image filenames

5. **Model Training** (`Resnet18.py`)
   - ResNet18 architecture with 6 output classes
   - Train/test split (80/20)
   - Balanced class weights to handle imbalance
   - Early stopping with 5-epoch patience
   - Batch size: 64, Epochs: 20

## Model Architecture

### ResNet18
- **Input Shape**: 28×28×3 (RGB images)
- **Initial Layer**: Conv2D (64 filters) + BatchNorm + ReLU + MaxPooling
- **Residual Blocks**: 8 blocks with 64, 128, 256, and 512 filters
- **Global Average Pooling**: Dimension reduction
- **Dropout**: 0.5 (overfitting prevention)
- **Output Layer**: Dense (6 units) + Softmax

### Key Features
- Skip connections for improved gradient flow
- Batch normalization for training stability
- Dropout regularization
- Balanced class weights for imbalanced data

## Requirements

```
numpy
tensorflow/keras
scikit-learn
opencv-python (cv2)
pillow
```

## Installation

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

## Usage

### 1. Load and Analyze Data
```bash
python Data_2.py
```
Outputs class distribution statistics.

### 2. Convert Vectors to Images
```bash
python Upload_Images.py
```
Converts training data vectors to PNG files for visualization and enhancement.

### 3. Enhance Images (Optional)
```bash
python Enhance_Images.py
```
Upscales images by 2x using cubic interpolation.

### 4. Convert Enhanced Images Back to Vectors
```bash
python Enhanced_Images_to_vectors.py
```
Creates `Xtrain_Classification2_enhanced.npy`.

### 5. Train and Evaluate Model
```bash
python Resnet18.py
```
- Trains ResNet18 on the dataset
- Evaluates on test split
- Generates predictions for test set
- Saves model and predictions to disk

## Performance Metrics

The model uses:
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy, Balanced Accuracy Score
- **Validation**: 20% held-out test set

## Files Generated

- `trained_model/`: Saved ResNet18 model weights
- `Ytest_Classification2.npy`: Final test predictions
- Enhanced images in `Images/enhanced/`

## Key Insights

- The model handles both dermoscopy and blood cell microscopy images in a unified framework
- Class imbalance is mitigated using computed class weights
- Early stopping prevents overfitting by monitoring validation loss
- The 28×28 image resolution is preserved to maintain training efficiency

## Future Improvements

- Experiment with larger image resolutions (using enhanced images)
- Implement data augmentation (rotation, flipping, brightness)
- Explore ensemble methods combining multiple models
- Add confidence scores and uncertainty estimates
- Implement explainability methods (Grad-CAM, LIME)

## Notes

- Image enhancement (2x upscaling) significantly increases training time and is not used in the final submission
- The model is optimized for the 28×28 input size
- Class weights are crucial for handling the significant class imbalance
- Early stopping patience is set to 5 epochs to avoid overfitting

## Author 
João Salbany

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Medical image datasets: Dermoscopy and Blood Cell Microscopy

---

**Last Updated**: March 2026
