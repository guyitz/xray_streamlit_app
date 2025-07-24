---
marp : true
---



# Course Project: X-ray Chest Image Classification for Medical Diagnosis


* **Elyahoo Reosenfeld, Guy Itzhaki**
* **Course Name:** Medical Imaging Systems
* **Date: 7/25**

---

## Project Goal & Initial Problem Statement

### 🎯 Project Goal
Develop an **automated system to classify X-ray chest images** into distinct medical conditions.

### ⚠️ The Initial Challenge
* Received 3 folders (01, 02, 03) of X-ray images.
* Each folder represented an **unknown medical condition**.

### 🚧 Crucial Limitations
* **Small Dataset:** Only ~200-300 images  This is **severely insufficient** for training complex Convolutional Neural Networks (CNNs) from scratch.
* **Limited Computing Power** 

---

## Overcoming Initial Limitations

### Problem: Unknown Conditions & Small Dataset

### Initial Approach: Leverage Pre-trained CNNs for Condition Identification
* Explored existing **CNN models pre-trained specifically on chest X-ray images** (e.g., from Hugging Face).
https://huggingface.co/codewithdark/vit-chest-xray

* By testing these models on our small initial datasets (folders 01, 02, 03), we inferred the medical conditions:
    * **Folder 01: Pneumonia**
    * **Folder 02: Normal**
    * **Folder 03: COVID-19**
---
### 🚀 Solution: Data Acquisition & Expansion
* With identified classes, we sourced publicly available, **larger chest X-ray datasets** for Pneumonia and COVID-19.
* **Result:** Acquired ~1.5 GB of additional images, significantly expanding our dataset:
    * **Pneumonia:** 5,618 images
    * **Normal:** 11,775 images
    * **COVID:** 3,616 images
* **Refined Classes:** Filtered out irrelevant classes (e.g., "lung opacity") to focus solely on our three target conditions (Pneumonia, Normal, COVID-19). This **improved model accuracy** by reducing noise and enhancing class discrimination.

---

## Data Preprocessing & Augmentation

Even with expanded data, deep learning models benefit from diverse training inputs to prevent overfitting and improve real-world generalization.

### ✨ Solution: Image Augmentation


* `transforms.Resize((224, 224))`: vgg imsage size
* `transforms.RandomHorizontalFlip(p=0.5)`: Flips images horizontally at random, improving model invariance to left-right orientation.
* `transforms.RandomRotation(15)`: Randomly rotates images by a small degree (up to 15), enhancing robustness to slight positional shifts.
* `transforms.RandomAffine(degrees=5, translate=(0.05, 0.05))`: Applies minor random rotations and translations, simulating real-world variations.
---
* `transforms.ToTensor()`: Converts images to PyTorch tensors, the required data format for model processing.
* `transforms.Normalize(...)`: Normalizes pixel values using ImageNet statistics, crucial for leveraging models pre-trained on ImageNet.


---

## Leveraging Pre-trained Models: Transfer Learning

###  Transfer Learning with Pre-trained CNNs
* We utilized powerful pre-trained models: **VGG16-BN** and **EfficientNet-B3**.
* These models were initially trained on the massive **ImageNet dataset**, containing millions of diverse images.

### 🔄 Transfer Learning Principle
* The **initial layers** of these pre-trained networks have learned to extract fundamental, generic image features 
* We then **fine-tune the higher-level layers** (and replace the classification head) to adapt these learned features to our specific task: classifying X-ray images into medical conditions. This significantly reduces the data and computational resources required compared to training from scratch.

---

## Model Selection and Architecture Comparison

### Chosen Models:

1.  **EfficientNet-B3**
    * **Why EfficientNet?** It was our primary choice due to its groundbreaking **"compound scaling" method**. EfficientNet intelligently and uniformly scales network depth, width, and resolution.
    * **Benefit:** This leads to state-of-the-art accuracy with **significantly fewer parameters and Floating Point Operations (FLOPs)** compared to other CNNs. This efficiency was a **perfect fit for our limited computing power**.
---
2.  **VGG16-BN (Batch Normalization)**
    * Chosen as a **comparative model**. It's a classic, deep CNN architecture known for its simplicity and strong performance.
    * **Why Batch Normalization (BN)?** The 'BN' variant includes Batch Normalization layers, which:
        * **Improve Training Stability:** Reduces "internal covariate shift," where the distribution of layer inputs changes during training.
        * **Accelerate Convergence:** Allows the use of higher learning rates, speeding up the training process.
        * **Act as a Regularizer:** Offers a slight regularization effect, sometimes reducing the need for aggressive dropout.

---

## Model Training Details: Fine-tuning Strategy

### 🛠️ Model-Specific Fine-tuning:

* **VGG16-BN:**
    * Initialized with ImageNet weights: `model = vgg16_bn(pretrained=True)`
    * **Unfroze the last 8 layers** of the convolutional `features` block. These layers (and their associated BatchNorm layers) had their weights updated.
    * The standard VGG classifier was **replaced** with a new `nn.Sequential` block, including a `Dropout` layer (to prevent overfitting) and a `Linear` layer (outputting to our 3 classes). This **new classifier head was trained from scratch**.
---
* **EfficientNet-B3:**
    * Initialized with ImageNet weights: `model = EfficientNet.from_pretrained('efficientnet-b3')`
    * **TorchScript Compatibility:** Replaced the custom `_swish` activation with `nn.SiLU` for better export compatibility.
    * The final fully connected (classifier) layer (`_fc`) was **replaced** with a new `nn.Sequential` block (including `Dropout` and `Linear` layers for our 3 classes). This **new classifier head was also trained from scratch**.
---
### 📊 Hyperparameter Optimization with Optuna

* To achieve optimal performance, we employed **Optuna**, an automatic hyperparameter optimization framework.
* Optuna intelligently searched for the best combinations of:
    * **Learning Rate (`lr`)**
    * **Batch Size (`batch_size`)**
    * **Dropout Rate (`dropout_rate`)**
    * **Number of Epochs (`num_epochs`)**
* This automated process maximized our model's validation accuracy, saving significant manual tuning effort.

---

## Experimental Setup & Data Splits

### 🧪 Data Strategy: Thorough Model Evaluation
We experimented by training and evaluating them on different dataset combinations:

1.  **Original Data Only:** Our initial, smaller dataset (after identifying classes).
2.  **Kaggle/Expanded Data Only:** The large public datasets we acquired.
3.  **Original + Kaggle Data:** The combined, largest dataset available.

---

### 📚 Dataset Split Ratios
For each data strategy, the dataset was rigorously split to ensure unbiased evaluation:

* **Training Set (60%):** 
* **Validation Set (20%):** 
* **Test Set (20%):** 

---

## Results & Discussion

* **VGG16-BN (Original + Kaggle Combined Data):**
    * **Accuracy:** **0.9657**
    * Precision (weighted): 0.9666
    * Recall (weighted): 0.9657
    * F1-Score (weighted): 0.9655

* **EfficientNet-B3 (Original + Kaggle Combined Data):**
    * **Accuracy:** **0.9822**
    * Precision (weighted): 0.9824
    * Recall (weighted): 0.9822
    * F1-Score (weighted): 0.9821


---

### 🔑 Key Findings & Insights

* **EfficientNet-B3 Outperformance:** EfficientNet-B3 consistently achieved **higher accuracy and superior metrics** compared to VGG16-BN. 
* **Impact of Data Expansion:** 
* **"Original Data Only" Nuance:**  **data quality can sometimes outweigh sheer quantity**.
* **Composite Model Decision:** We considered developing an ensemble model (combining predictions from multiple models). However, given the **outstanding performance of the EfficientNet-B3 model on its own**, a composite model was not pursued, as it would introduce additional complexity without a significant, demonstrable gain in accuracy.

---

## Conclusion 

### ✅ Conclusion

* Successfully developed a **robust X-ray chest image classification system** capable of distinguishing between Pneumonia, Normal, and COVID-19 cases with high accuracy.
* Demonstrated the effectiveness of **transfer learning** and **strategic data augmentation** in overcoming significant limitations of small datasets and constrained computing power.
* **EfficientNet-B3** proved to be an excellent choice for this medical imaging task, excelling in both accuracy and computational efficiency.
* Accurate classification of medical conditions achieved with high confidence, showcasing the potential of deep learning in clinical settings.
