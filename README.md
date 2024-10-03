# Blood Vessel Segmentation - 3D Medical Imaging Project

This repository contains my work for the Kaggle competition titled **"SenNet + HOA - Hacking the Human Vasculature in 3D"**. The goal of this project is to develop a robust machine learning model capable of segmenting blood vessels from 3D medical imaging data of human kidneys. Below, I describe the dataset, the preprocessing steps, the model architecture, training methodology, and results (Project originally completed in December 2023. Uploaded to GitHub in October 2024 for portfolio purposes).

## Project Overview

Blood vessel segmentation in 3D medical imaging is a challenging task that plays a crucial role in understanding vascular structures and improving diagnostic outcomes. The project utilizes a dataset consisting of high-resolution 3D scans of human kidney tissue, with corresponding segmentation masks. Using a U-Net-based deep learning model, I aimed to automate the identification and segmentation of vascular structures, significantly enhancing diagnostic efficiency and accuracy in medical imaging.

## Dataset

- **Source:** The dataset for this project was sourced from the Kaggle competition consists of over 14365 files (43.52 GB). You can access and download the dataset from Kaggle using the following link: https://www.kaggle.com/competitions/blood-vessel-segmentation/data
- **Content:** 
  - High-resolution 3D medical images of kidney tissues in TIFF and NRRD formats.
  - Corresponding binary segmentation masks indicating blood vessel regions.
- **Preprocessing:**
  - **Normalization:** The intensity values were normalized to a [0, 1] range to ensure consistency across different scanning conditions.
  - **Mask Resizing:** Masks were resized to match the downsampled resolution of the images using nearest-neighbor interpolation to prevent partial-class artifacts.
  - **Binarization:** A binarization step was applied to ensure each pixel was strictly classified as either a vessel or non-vessel.

## Data Splitting Strategy

The dataset was stratified based on vascular density to ensure that the model trained and validated on a representative sample, covering a wide range of vascular complexities. This approach helps in improving model generalization on diverse unseen data.

## Model Architecture

- **Model Type:** I implemented a **U-Net** architecture, which is well-known for its effectiveness in medical image segmentation tasks. The U-Net model was designed with:
  - **Contracting Path:** Series of convolutional layers with batch normalization and max-pooling to down-sample the feature maps.
  - **Expanding Path:** Up-sampling layers to recover the spatial resolution and concatenate with feature maps from the contracting path for accurate localization.
  - **Bottleneck Layer:** Introduced at the deepest part of the network to capture global features, followed by dropout to prevent overfitting.
  - **Final Layer:** A 1x1 convolution to predict the segmentation mask for blood vessel regions.

- **Optimizer & Loss Function:** The model was compiled with the **Adam optimizer** and **binary cross-entropy loss function**.
- **Custom Metric:** A custom **Dice coefficient** metric was used to evaluate the model's segmentation accuracy. The Dice coefficient is a statistical measure of overlap, providing an intuitive assessment of model performance.

## Training

- **Training Hyperparameters:**
  - **Epochs:** 30
  - **Batch Size:** 8
  - **Learning Rate:** Adaptive via the Adam optimizer.
  - **Regularization:** Batch normalization and dropout layers were included to stabilize training and prevent overfitting.
  
- **Results:**
  - The model achieved a **Dice coefficient of approximately 0.80** on the validation set, significantly outperforming manual segmentation and other baseline methods.

## Benchmarking

The performance of my U-Net model was benchmarked against other segmentation methods:

- **3D U-Net:** A comparison with 3D U-Net showed that while the latter manages volumetric data effectively, its computational demands were much higher. My approach using 2D slices achieved similar accuracy with better computational efficiency.
- **Random Forest Classifiers:** My model surpassed traditional Random Forest classifiers, which struggled with the high-dimensional nature of 3D medical imaging data.
- **2D Segmentation Methods:** By training on sequential slices, my U-Net approximated 3D context, providing a more accurate understanding of vascular structures compared to isolated 2D methods.

## Key Techniques and Challenges

- **Stratified Dataset Splitting:** Addressing variability in vascular structures across different individuals was crucial for model robustness.
- **Normalization and Mask Binarization:** Consistent input preprocessing played a significant role in ensuring high-quality learning during training.
- **Balancing Accuracy and Efficiency:** The 2D U-Net implementation effectively balanced the need for computational efficiency with the accuracy required for medical diagnostics.

## Conclusion

This project has been an insightful journey in applying deep learning techniques to a real-world medical imaging problem. By successfully implementing a U-Net architecture and achieving a competitive Dice score, the model demonstrates its potential for improving diagnostic workflows. The use of advanced preprocessing, data stratification, and careful benchmarking highlights the robustness of the solution.

## Repository Contents

- **Notebook (`Yixuan Zheng - blood_vessel_segmentation.ipynb`):** Contains all the code related to data preprocessing, model implementation, training, and evaluation.
- **README (`README.md`):** Provides an in-depth explanation of the project, including background, methodology, model architecture, results, and insights gained.

## How to Use This Repository

1. Clone the repository:
   ```sh
   git clone https://github.com/yixuan-zheng/blood-vessel-segmentation.git
   ```
2. Navigate to the project directory:
   ```sh
   cd blood-vessel-segmentation
   ```
3. Open the Jupyter notebook to explore the code and reproduce the results:
   ```sh
   jupyter notebook "Yixuan Zheng - blood_vessel_segmentation.ipynb"
   ```

## Future Work

- **3D U-Net Implementation:** As part of future development, implementing a full 3D U-Net architecture could provide more contextual information.
- **Further Optimization:** Exploring more advanced optimization techniques to further reduce training time while maintaining or improving accuracy.
- **Clinical Testing:** Moving towards clinical validation by collaborating with healthcare professionals to assess real-world performance and usability.
