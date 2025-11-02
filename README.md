# Speech Emotion Recognition using Deep Learning

## Table of Contents
- [About the Project](#about-the-project)
- [Objectives & Scope](#objectives--scope)
- [Dataset Description](#dataset-description)
- [Project Workflow](#project-workflow)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Kaggle Notebook](#Kaggle-notebook)
- [Preprocessing Steps](#preprocessing-steps)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Technologies Used](#technologies-used)
- [Installation and Setup](#installation-and-setup)
- [Usage Instructions](#usage-instructions)
- [Contributions](#contributions)
- [Project Structure](#project-structure)
- [Challenges & Lessons Learned](#challenges--lessons-learned)
- [Proposal Report](#proposal-report)
- [Citations & References](#citations--references)
- [Acknowledgments](#acknowledgments)
- [Screenshots or Demo](#screenshots-or-demo)

## About the Project
This project focuses on **Speech Emotion Recognition (SER)**, which is an intersection of speech processing, emotion psychology, and deep learning. The primary objective is to teach machines to detect and understand human emotions based on spoken language. By analyzing emotional states through sound features, we model these emotions using neural networks. The goal is to create a system that can interpret emotions just by listening to speech, making machines more empathetic in communication.

## Objectives & Scope
The objectives of this project include:
- **Emotion Detection from Speech**: Automatically recognizing emotions from spoken language to improve human-computer interaction.
- **Real-World Applications**: The technology is applicable in **customer service automation**, **virtual assistants**, and **mental health monitoring**.
- **Challenges Faced**: The project works to overcome challenges such as recognizing subtle emotional variations, dealing with noise in the dataset, and ensuring the model generalizes well across various emotional states.

## Dataset Description
The dataset used is the **TESS dataset**, which includes recordings from two female speakers: one representing an older adult female (OAF) aged around 64 years, and the other a younger adult female (YAF) aged around 26 years. Each speaker recorded sentences in different emotional tones, covering seven basic emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**. The dataset contains multiple recordings for each emotion, providing diverse vocal expressions.

## Project Workflow
The workflow involves the following steps:
1. **Data Collection**: Gathering the audio files from directories.
2. **Preprocessing**: Extracting emotion labels and performing feature extraction.
3. **Model Training**: Building a deep learning model for emotion recognition.
4. **Evaluation**: Assessing the model’s performance on unseen data.

## Key Features
- **Hybrid CNN-LSTM Model**: Combines Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) networks to capture temporal dependencies.
- **Emotion Recognition from Speech**: The model can classify seven emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.
- **High Performance**: The model achieves strong performance metrics, including 97.65% training accuracy and 96.90% test accuracy.
- **Efficient Data Preprocessing**: The pipeline extracts features from audio files using **Librosa** and efficiently handles the dataset with **Pandas**.

## Model Architecture
The model utilizes a **hybrid CNN-LSTM architecture**:
- **CNN Layers**: Used for spatial feature extraction from audio features (MFCCs).
- **LSTM Layers**: Capture temporal dependencies in speech signals, enabling the model to understand the sequence of speech data.
  
## Kaggle Notebook
[Open kaggle notebookt](Speech%20emotion%20recognition%20analysis%20using%20deep%20learning%20.ipynb)

## Preprocessing Steps
The preprocessing steps include:
1. Using `os.walk()` to traverse directories and collect audio file paths.
2. Extracting emotion labels from file names and converting them for supervised learning.
3. Loading a subset of **2,800 audio files** to optimize memory usage.
4. Creating a **Pandas DataFrame** with 'speech' and 'label' columns to manage the dataset.
5. Inspecting the dataset for class distribution to address imbalances.

## Training and Evaluation
- **Training Process**: The model is trained with a batch size of 32 over 100 epochs, using **early stopping** as a callback to avoid overfitting.
- **Performance Metrics**:
  - **Training Accuracy**: 97.65%
  - **Validation Accuracy**: 97.38%
  - **Test Accuracy**: 96.90%
  - **F1-Score**: 0.9686 (balanced performance across emotion classes)

## Results
- **Training Accuracy**: 97.65%
- **Validation Accuracy**: 97.38%
- **Test Accuracy**: 96.90%
- **F1-Score**: 0.9686 (balanced performance across all emotion classes)
- **Test Loss**: 0.1117
- **Class Performance**: **Angry** (98.33%) and **Sad** (96.67%) had high accuracy, while **Ps** (83.33%) had the lowest.

## Limitations and Future Work
- **Dataset Limitations**: The TESS dataset is limited to 7 emotions and primarily includes English speakers, which may affect model generalization to other languages or emotional expressions.
- **Class Imbalance**: Some emotions have more samples than others, potentially biasing the model.
- **Future Work**:
  - **Cross-Lingual Emotion Recognition**: Train the model on a more diverse dataset to recognize emotions across languages.
  - **Real-Time Emotion Detection**: Implement the model for real-time emotion recognition.
  - **Multi-Modal Emotion Recognition**: Combine audio with other modalities (e.g., facial expressions) to improve accuracy.

## Technologies Used
- **Python**: The primary programming language for the project.
- **NumPy**: For numerical computations and handling multi-dimensional arrays.
- **Pandas**: For efficient data manipulation and preprocessing.
- **Seaborn & Matplotlib**: For visualizations and plotting.
- **Librosa**: For audio feature extraction.
- **TensorFlow/Keras**: For deep learning model implementation.

## Installation and Setup
1. Install the required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn librosa





## Usage Instructions
1. **Preprocessing**: Run the preprocessing script to load and preprocess the audio data.

2. **Training**: Execute the training script to train the model.

3. **Evaluation**: Evaluate the model's performance using the provided evaluation scripts.

4. **Visualizations**: Review the results in the **Screenshots or Demo** section (graphs, plots, and other performance metrics). These will provide insights into the model's accuracy, prediction confidence, and performance across datasets.


## Contributions
Contributions are welcome! To contribute:
1. **Fork the repository** to your own GitHub account.
2. Create a **pull request** with your proposed changes or improvements.
3. You can help improve the model by:
   - Experimenting with different model architectures or hyperparameters.
   - Improving documentation for better clarity.
   - Reporting any bugs or issues you encounter.
   
   Please make sure to follow the guidelines for contributing provided in the repository.


## Project Structure
The project is organized as follows:
- **data/**: Contains the audio files used for training the model.
- **notebooks/**: Jupyter notebooks with code for data preprocessing, feature extraction, and model evaluation.
- **src/**: Python scripts for preprocessing, training, and evaluating the model.
- **requirements.txt**: Contains a list of required Python packages and dependencies to set up the environment.

---

## Challenges & Lessons Learned
- **Class Imbalance**: The dataset has some class imbalances. This was addressed through techniques such as **data augmentation** and adjusting class weights during training.
- **Model Convergence**: Finding the optimal learning rate and batch size was critical to achieving fast convergence and good model performance.
- **Feature Selection**: The feature extraction process using **MFCCs** and **spectrograms** played a crucial role in the accuracy of the emotion recognition task.

---
## Proposal Report 
- [Open Project Report](final%20proposal.pdf)

---

## Citations & References
- [Open Literature review Report](literature%20review.pdf)

---

## Acknowledgments
- We would like to express our deepest gratitude to our honorable teacher, Mahbuba Habib Ma’am, Lecturer, Department of Computer Science and Engineering, Bangladesh University of Business and Technology (BUBT), for her invaluable guidance, continuous support, and constant motivation throughout the completion of this work. Her insightful suggestions and encouragement have been instrumental in the successful development of this project.

---

## Screenshots or Demo
This section will include the following visualizations:
1. **Performance Metrics Comparison**: A graph comparing training, validation, and test accuracies over epochs.
2. **Prediction Confidence Distribution**: A graph showing the confidence of the model's predictions across different emotions.
3. **Performance Across Datasets**: Comparative performance graphs for different datasets.
4. **Model Architecture Summary**: Diagram of the CNN-LSTM model showing the layers and their functions.

*(Will update later . need to take some screenshoots .)*

---
