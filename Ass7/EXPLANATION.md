# Assignment 7: Transfer Learning (AlexNet, VGG16, ResNet50, EfficientNetB0)

## 🎯 Objective
Implement and benchmark **Transfer Learning** using four standard CNN architectures. We freeze the deep convolutional layers pre-trained on ImageNet and retrain only the final dense classifier head.

## 🧠 Concepts Covered
- **Pre-trained Models**: Extracting spatial features learned from millions of ImageNet images.
- **Layer Freezing**: Disabling gradient tracking (`requires_grad=False`) to preserve learned weights and drastically speed up training.
- **Architecture Variations**:
  - `AlexNet` (Historical pioneer)
  - `VGG16` (Deep & Uniform 3x3)
  - `ResNet50` (Residual / Skip Connections)
  - `EfficientNetB0` (Compound scaling optimization)
- **Image Normalization**: Matching the input tensor statistics (Mean and Std. Deviation) expected by PyTorch pre-trained weights.

## 📁 Folder Structure
- `Assignment7.ipynb`: Core implementation notebook (PyTorch).
- `60_Assignment-7_DL.docx`: Submission-ready lab report.
- `Datasets/`: Target folder for downloaded dataset artifacts.
- `Plots/`: Contains benchmark visualization `transfer_learning_comparison.png`.
