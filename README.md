# 🖼️ Image Captioning with CNN-LSTM

This project demonstrates a basic implementation of **Image Captioning** using a hybrid **Convolutional Neural Network (CNN)** and **Long Short-Term Memory (LSTM)** model. It was developed as part of an in-class hands-on session with guidance from the lecturer.

The goal of this project is to generate natural language captions for images by combining computer vision and natural language processing techniques.

---

## 🎯 Objectives

- Understand the workflow of combining CNN (for feature extraction) and LSTM (for sequence modeling).
- Extract visual features from images using a pre-trained CNN (e.g., InceptionV3 or VGG16).
- Use LSTM to generate text descriptions based on extracted image features.
- Train the model on a subset of a captioning dataset (e.g., Flickr8k or COCO).
- Evaluate model performance and observe generated captions.

---

## 🧰 Tools & Libraries

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- NLTK (for text preprocessing)
- Pre-trained CNN models (e.g., InceptionV3)

---

## 🗃 Workflow Overview

1. **Data Preprocessing**
   - Load image-caption pairs.
   - Clean and tokenize captions.
   - Create word-to-index and index-to-word mappings.
   - Pad sequences for uniform input length.

2. **Feature Extraction**
   - Load images and pass them through a pre-trained CNN.
   - Save the intermediate feature vectors (without the classification head).

3. **Model Architecture**
   - CNN feature vector input + Embedding + LSTM + Dense layers.
   - Caption generation trained using teacher forcing.

4. **Training**
   - Train on (image features, partial caption) → next word.
   - Monitor loss and accuracy.

5. **Caption Generation**
   - Generate captions for new images by predicting words one at a time.
   - Stop when the model predicts an `<end>` token.

---

## 📌 Notes

- This is a learning-focused project, not a production-ready model.
- Code and dataset used may be simplified for instructional purposes.

---

## 👩‍🏫 Acknowledgment

This project was completed as part of an in-class practical session.

---

## 📄 License

This project is for educational purposes and shared under the MIT License.
