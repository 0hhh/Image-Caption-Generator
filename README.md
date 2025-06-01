# 🖼️ Image Caption Generator

A deep learning project to automatically generate captions for images using Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) with attention to data cleaning, feature extraction, and sequence modeling.

---

## 🚀 Project Workflow

1. **Data Preparation**
    - **Load Captions:** Read image-caption pairs from the dataset.
    - **Clean Captions:** Lowercase, remove punctuation, filter out non-alphabetic words.
    - **Build Vocabulary:** Extract all unique words from the cleaned captions.
    - **Save Cleaned Captions:** Store processed captions for later use.

2. **Feature Extraction**
    - **CNN Model:** Use Xception (pre-trained on ImageNet) to extract features from each image.
    - **Save Features:** Store extracted features in a pickle file for efficient loading.

3. **Data Loading**
    - **Load Training Images:** Read the list of training images.
    - **Load Cleaned Captions:** Filter captions for only the training images.
    - **Load Features:** Load features for only the training images.

4. **Text Tokenization**
    - **Tokenizer:** Fit a Keras tokenizer on all training captions.
    - **Save Tokenizer:** Store the tokenizer for inference and reproducibility.
    - **Calculate Max Length:** Find the maximum caption length for padding.

5. **Sequence Preparation**
    - **Create Sequences:** Convert each caption into input-output pairs for training (image features, input sequence, output word).
    - **Data Generator:** Use a TensorFlow data generator to efficiently feed data to the model.

6. **Model Architecture**
    - **Image Model:** Dense layers to process CNN features.
    - **Text Model:** Embedding and LSTM layers to process caption sequences.
    - **Merge:** Combine both models and output a word prediction.
    - **Compile:** Use categorical cross-entropy loss and Adam optimizer.

7. **Training**
    - **Epoch Loop:** For each epoch:
        - Generate data batches.
        - Train the model for a set number of steps.
        - Save the model after each epoch.

---

## 📁 Directory Structure

Image-Caption-Generator/

├── main.py  
├── features.p  
├── tokenizer.p  
├── descriptions.txt  
├── models/  
│   ├── model_0.h5  
│   ├── model_1.h5  
│   └── ...  
├── Flicker8k_Dataset/  
├── Flickr8k_text/  
└── README.md  


## ⚙️ Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Pillow
- tqdm
- matplotlib
  

## 🏁 Getting Started
Download the Flickr8k dataset and place images and text files in the appropriate folders.  
Run main.py to preprocess data, extract features, and train the model.  
Generated models will be saved in the models/ directory.  

## ✨ Results
After training, you can use the saved models and tokenizer to generate captions for new images!
