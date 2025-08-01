# Report: Chest X-ray Image Classification for Medical Diagnosis

**Eliyahu Rosenfeld – 324462514**
**Guy Itzhaki – 60877693**

## **1. Introduction**

This report outlines the development of an automated system for classifying chest X-ray images. The primary objective of the project was to accurately classify X-ray images into three distinct classes.
In the project, we made an "educated guess" that these classes were: Pneumonia, Normal, and COVID-19, respectively. We leveraged deep learning techniques, specifically transfer learning with pre-trained Convolutional Neural Networks (CNNs), to achieve this goal.
We also conducted many experiments—such as training VGG-16 without batch normalization and testing machine learning models like Random Forest (which actually performed quite well).
However, only the best-performing models are presented in this report.

## **2. Preparing the Data**

### **2.1 Data Acquisition and Expansion**

The initial dataset was very limited, consisting of approximately 300 images across three unlabeled folders.
To address this, our first step was to identify the medical conditions represented in these folders using pre-trained models. This allowed us to label the folders as "Pneumonia," "Normal," and "COVID-19."

We made these guesses by running the images through trained models found on Hugging Face.
Additionally, we consulted medical websites that explain how to diagnose chest X-ray images.

Once the classes were identified, we expanded the dataset by sourcing larger, publicly available datasets from platforms like Kaggle, increasing our image count significantly:

* **Pneumonia**: 5,618 images
* **Normal**: 11,775 images
* **COVID-19**: 3,616 images

(We are unsure how "educated" these guesses were.)

### **2.2 Data Preprocessing and Augmentation**

To prepare the images for training and reduce overfitting, we applied several preprocessing and augmentation steps using `torchvision.transforms`:

* `transforms.Resize((224, 224))`: Resized all images to 224x224 pixels, the input size required by VGG16.
  (224 for VGG16 and 300 for EfficientNet)
* `transforms.RandomHorizontalFlip(p=0.5)`: Randomly flipped images horizontally to introduce left-right invariance.
* `transforms.RandomRotation(15)`: Applied small random rotations to increase robustness to position shifts.
* `transforms.RandomAffine(degrees=5, translate=(0.05, 0.05))`: Applied slight affine transformations to simulate real-world imaging variation.
* `transforms.ToTensor()`: Converted images to PyTorch tensors.
* `transforms.Normalize(...)`: Normalized pixel values using ImageNet's mean and standard deviation. This step is essential for compatibility with models pre-trained on ImageNet.

### **2.3 Dataset Split**

The dataset was divided into three subsets to ensure an unbiased evaluation:

* **Training Set (60%)**: Used to train the model.
* **Validation Set (20%)**: Used for tuning and to monitor overfitting.
* **Test Set (20%)**: Held out for final evaluation only.

## **3. Choosing a Model**

### **3.1 Transfer Learning**

Given the initial dataset size and limited computing resources, we used transfer learning. This involved leveraging powerful pre-trained models that had already learned to extract general features (like edges and textures) from the ImageNet dataset.
We fine-tuned the top layers of these models and replaced the classification head to suit our three-class task.

Additionally, we trained our models on AWS machines, since the free Colab GPU (limited to 4 hours) was insufficient.
We used the `g5.2xlarge` instance ([link](https://aws.amazon.com/ec2/instance-types/g5/)) for training.

### **3.2 Chosen Architectures**

We selected two state-of-the-art CNN architectures:

1. **VGG16-BN (Batch Normalization)**: A classic deep CNN with batch normalization layers for better training stability and faster convergence.

2. **EfficientNet-B3**: A modern, efficient architecture that uses compound scaling. It achieves high accuracy with fewer parameters, making it ideal for resource-limited environments.

EfficientNet gave the best results. After reading many articles recommending EfficientNet for X-ray classification, we invested time in finding a CNN that could outperform VGG.

## **4. Training the Model**

*(The training process described here applies to VGG16-BN; EfficientNet training was very similar.)*

* **Model Architecture**: We loaded VGG16-BN with ImageNet weights and unfroze the last 8 convolutional layers for fine-tuning. The classifier head was replaced with a new `nn.Sequential` block tailored for our 3-class task.
* **Loss Function**: `nn.CrossEntropyLoss()` was used for multi-class classification.
* **Optimizer**: Adam optimizer with weight decay of 5e-5.
* **Scheduler**: `ReduceLROnPlateau` reduced the learning rate when validation accuracy plateaued.
* **Early Stopping**: Training stopped early if validation accuracy didn’t improve for 5 epochs.

## **5. Parameter Tuning**

We used **Optuna**, an automatic hyperparameter optimization tool, to tune:

* **Learning Rate**: 1e-5 to 1e-2
* **Batch Size**: 8, 16, 32
* **Dropout Rate**: 0.3 to 0.5
* **Number of Epochs**: 5 to 15

(*We used ChatGPT for initial tuning suggestions, and improved based on observed results.*)

The best parameter combination found by Optuna was used to train the final VGG16-BN model.

## **6. Evaluating the Model**

The final trained model was evaluated on the test set (20% of data, not used in training or validation).

### **6.1 Test Results**

**VGG16-BN**

* Accuracy: 0.9657
* Precision (weighted): 0.9666
* Recall (weighted): 0.9657
* F1-Score (weighted): 0.9655

**EfficientNet-B3**

* Accuracy: 0.9822
* Precision (weighted): 0.9824
* Recall (weighted): 0.9822
* F1-Score (weighted): 0.9821


