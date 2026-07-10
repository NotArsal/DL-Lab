## CI3201: Deep Learning

**Credits:** 4  
**Teaching Scheme:**  
- **Theory:** 3 Hours/Week  
- **Lab:** 2 Hours/Week  

**Course Prerequisites:** Machine Learning, Python Programming, Linear Algebra, Probability and Statistics, Full Stack Development

### Course Objectives:
1. To introduce the fundamental concepts, architectures, and tools used in deep learning.
2. To develop an understanding of neural networks, activation functions, and learning algorithms.
3. To study deep learning architectures such as CNN, RNN, and GRU for processing various types of data.
4. To explore convolutional neural network architectures and techniques for image representation and visual recognition tasks.
5. To learn modern deep learning techniques including transfer learning, autoencoders, transformers, and optimization methods.
6. To apply deep learning models for solving real-world problems in image, text, audio, video, and time-series domains.

### Course Relevance:
This course focuses on the design, training, and optimization of deep neural network architectures for learning complex patterns from large-scale data. It equips students with techniques such as CNNs, RNNs, LSTMs, Transformers, and transfer learning for solving advanced AI problems. The course forms the foundation for developing state-of-the-art solutions in computer vision, natural language processing, speech analytics, and generative AI applications.

---

### Section 1: Topics/Contents

**Unit-I: Fundamental of Deep Learning (06 Hours)**
Introduction to Artificial Intelligence, Machine Learning, and Deep Learning, Limitations of machine learning, Advantage and challenges of deep learning, Evolution and applications of Deep Learning, Introduction to TensorFlow, Keras, and PyTorch, GPU and cloud-based platforms (Google Colab), Overfitting and Underfitting, Regularization Techniques, Batch Processing and Epochs, Data Preparation for Deep Learning: Dataset collection and preprocessing, Data normalization and scaling, Training, validation, and testing datasets.

**Unit-II: Perceptron and Neural Network Architecture (06 Hours)**
Biological neuron vs artificial neuron, Perceptron model and multilayer perceptron, Weight and Bias, Activation functions: Sigmoid, Tanh, ReLU, Softmax, Loss functions and optimization basics, Notation, Calculation of Trainable Parameters, Backpropagation and Forward propagation, XOR Problem and Limitations of Perceptron.

**Unit-III: RNN, LSTM and GRU Architectures (09 Hours)**
Introduction to Sequential Data, Need for Recurrent Neural Networks, Applications of RNN in Real-world Problems, Architecture of Recurrent Neural Network, Difference Between Feed Forward Network and RNN, Layers in RNN, Mathematical Representation of RNN, Forward Propagation, Backpropagation Through Time (BPTT), Training Process, Vanishing Gradient Problem, Exploding Gradient Problem, Limitations of Traditional RNN. Introduction to Long Short-Term Memory (LSTM), Architecture, Gates in LSTM, Mathematical Equations of LSTM Gates, Calculation of Trainable Parameters in RNN and LSTM, Applications of LSTM in NLP and Time Series Forecasting, Gated Recurrent Unit (GRU), Need for GRU, Architecture, Limitations of Traditional RNN, Architecture of GRU, Components of GRU, Comparison Between RNN, LSTM, and GRU.

---

### Section 2: Topics/Contents

**Unit-IV: Convolutional Neural Networks (CNN) (06 Hours)**
Introduction to CNN: Need for CNN in image processing, Types of CNN (1D CNN, 2D CNN, and 3D CNN), Convolution operation and feature extraction, CNN Architecture: Convolution layer, Filters, Kernels, Stride, and Padding.
Activation Functions in CNN, Pooling Layers (Max Pooling, Average Pooling), Fully Connected Layers, Batch Normalization and Dropout, CNN Hyperparameters. Popular CNN Models: AlexNet, VGG, ResNet, EfficientNet, Applications of CNN.

**Unit-V: Advanced Deep Learning Architectures (07 Hours)**
Limitations of Traditional Neural Networks, Introduction to Transformer Architecture: Motivation, Encoder–Decoder Architecture, Self-Attention Mechanism, Multi-Head Attention, Positional Encoding, Advantages of Transformers. Transfer Learning: Pre-trained models and fine-tuning, Autoencoders and Generative Models: Autoencoders and dimensionality reduction, Bidirectional Encoder Representations (BERT), Generative Adversarial Networks (GANs), Graph Neural Networks (GNN), Vision Transformers (ViT).

**Unit-VI: Applications of Deep Learning (08 Hours)**
Deep Learning for Image Processing: Image segmentation and enhancement, Disease prediction using medical images. Deep Learning for Text Processing: Natural Language Processing (NLP), Chatbots, Text classification and summarization. Deep Learning for Audio and Video Processing: Speech-to-text systems, Audio classification, Video analysis. Deep Learning for Numerical and Sensor Data: Time-series prediction, financial forecasting, IoT and smart sensor applications.

---

### List of Practicals:
1. Install and configure TensorFlow/Keras in Google Colab. Perform data preprocessing, normalization, train-test splitting, and data visualization on a sample dataset.
2. Design and implement a Multilayer Perceptron (MLP) for classification of the Iris or Wine dataset, and evaluate its performance using accuracy and a confusion matrix.
3. Implement forward propagation and backpropagation using TensorFlow/Keras. Analyze the effect of different learning rates and the number of epochs on model performance.
4. Develop an LSTM-based model for time-series forecasting using stock price, weather, or sales datasets.
5. Implement and compare RNN, LSTM, and GRU models for sequence classification, and analyze their performance using appropriate evaluation metrics.
6. Design and implement a Convolutional Neural Network (CNN) for image classification using the Tomato or Soybean disease dataset.
7. Implement transfer learning using pre-trained AlexNet, VGG16, ResNet50, and EfficientNetB0 models for image classification, and compare their performance.
8. Implement a pre-trained BERT model for sentiment analysis or text classification on a sample dataset.
9. Perform image classification using a pre-trained Vision Transformer (ViT) model and compare its performance with a CNN model.
10. Design and implement a deep learning application for audio classification that incorporates Speech-to-Text (STT) and Text-to-Speech (TTS) using an appropriate deep learning architecture, and evaluate its performance using suitable metrics.

---

### List of Course Project Areas:
1. Plant Disease Detection and Severity Assessment System: Develop a deep learning model to identify plant diseases and estimate disease severity from leaf images.
2. AI-Based Brain Tumor Classification Using MRI Images: Design a transfer learning-based system to classify brain MRI scans into different tumor categories.
3. Fake News Detection Using Deep Learning and NLP: Develop an intelligent system that classifies news articles as genuine or fake using text analysis.
4. Deepfake Image and Video Detection Framework: Build a deep learning model to detect manipulated images and videos using visual artifacts.
5. Speech Emotion Recognition System: Design a model to recognize human emotions from speech recordings using deep learning techniques.
6. Image Caption Generation Using CNN and LSTM: Develop a system that automatically generates meaningful captions for input images.
7. Intelligent Traffic Density Estimation and Signal Optimization: Create a deep learning-based traffic analysis system to estimate vehicle density and optimize signal timing.
8. Human Activity Recognition Using Wearable Sensor Data: Develop an LSTM-based system to identify human activities from smartphone or IoT sensor data.
9. Sign Language Recognition Using Deep Learning: Design a vision-based system that converts sign language gestures into text or speech.
10. Multilingual Chatbot Using Transformer Models: Develop an AI chatbot capable of understanding and responding in multiple languages.
11. Audio-Based Bird Species Classification System: Build a deep learning model to identify bird species from audio recordings of bird calls.
12. Network Intrusion Detection Using Deep Neural Networks: Develop a cybersecurity system that detects malicious network activities using deep learning.
13. Automatic Medical Report Classification and Summarization: Design an NLP-based system to categorize and summarize medical reports automatically.
14. Financial Fraud Detection Using Deep Learning: Develop a predictive model to identify fraudulent financial transactions from historical data.
15. AI-Based Resume Screening and Candidate Ranking System: Create a deep learning system that analyzes resumes and ranks candidates based on job requirements.
16. Image Forgery Detection Using CNN and Attention Mechanisms: Develop a model to detect tampered or manipulated digital images.
17. Automatic Waste Classification for Smart Cities: Design a deep learning-based waste segregation system for recyclable and non-recyclable materials.
18. Cervical Cancer Cell Classification Using Transfer Learning: Design a deep learning framework for automated classification of cervical cancer cell images.

---

### Assessment Scheme:
- **Course Project:** End Semester Examination: 100 Marks converted to 30 Marks
- **Laboratory and Comprehensive Viva Voce:** End Semester Examination: 100 Marks converted to 30 Marks
- **Theory:** End Semester Examination: 60 Marks converted to 40 Marks

**Assessment scheme covers following aspects of Modified Blooms Taxonomy:**
L2 Understanding, L3 Apply, L3 Design, L3 Apply, L4 Analyze and L5 Evaluate

---

### Text Books: (As per IEEE format)
1. N. Buduma, N. Buduma, and J. Papa, Fundamentals of Deep Learning, 2nd ed. Sebastopol, CA, USA: O'Reilly Media, 2022.
2. C. C. Aggarwal, Neural Networks and Deep Learning. Cham, Switzerland: Springer, 2018.
3. F. Chollet, Deep Learning with Python. Shelter Island, NY, USA: Manning Publications, 2018.

### Reference Books: (As per IEEE format)
1. O. Campesato, Artificial Intelligence, Machine Learning, and Deep Learning. Dulles, VA, USA: Mercury Learning and Information LLC, 2020.
2. S. Weidman, Deep Learning from Scratch: Building with Python from First Principles, 1st ed. Sebastopol, CA, USA: O'Reilly Media, 2019.
3. Géron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd ed. Sebastopol, CA, USA: O'Reilly Media, 2019.
4. Zhang, Z. C. Lipton, M. Li, and A. J. Smola, Dive into Deep Learning. Cambridge, U.K.: Cambridge University Press, 2023.

### MOOCs Links and additional reading material:
1. https://nptel.ac.in/courses/106106184 [Deep Learning - IIT Ropar]
2. https://www.coursera.org/specializations/deep-learning [Deep Learning Specialization]
3. https://www.udemy.com/course/deeplearning/?couponCode=MT260629G2 [Deep Learning A-Z [2026]: DL, AI in Python & AWS + LLM Prize]
4. https://www.edx.org/learn/deep-learning [Learn deep learning]
5. https://course.fast.ai/ [Practical Deep Learning]

---

### Course Outcomes:
After completion of the course, student will be able to:
1. Explain the fundamentals of deep learning, data preparation techniques, and deep learning frameworks for building intelligent systems.
2. Apply perceptron and neural network architectures, activation functions, and backpropagation algorithms for solving classification problems.
3. Examine sequential data using RNN, LSTM, and GRU architectures for prediction and classification tasks.
4. Analyze the architecture, components, and performance of convolutional neural network models for image processing and computer vision applications.
5. Apply advanced deep learning architectures and optimization techniques to solve complex learning problems across diverse application domains.
6. Implement deep learning solutions for real-world applications involving image, text, audio, video, and numerical data.

### CO-PO Map:

| CO/PO | PO1 | PO2 | PO3 | PO4 | PO5 | PO6 | PO7 | PO8 | PO9 | PO10 | PO11 | PO12 | PSO1 | PSO2 | PSO3 |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|------|------|------|------|------|
| **CO1** | 2 | 2 | 1 | - | - | - | - | - | - | - | - | - | 2 | 2 | - |
| **CO2** | 3 | 2 | 2 | - | 2 | - | - | - | - | - | - | 2 | 3 | 3 | 2 |
| **CO3** | 3 | 3 | 2 | 2 | 2 | - | - | - | - | - | - | 2 | 3 | 3 | 2 |
| **CO4** | 3 | 3 | 2 | 2 | 2 | - | - | - | - | - | - | 2 | 3 | 3 | 2 |
| **CO5** | 3 | 3 | 2 | 2 | 2 | 2 | 2 | - | - | - | - | 2 | 3 | 3 | 2 |
| **CO6** | 3 | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | 2 |
| **Average** | 2.83 | 2.67 | 1.83 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.83 | 2.83 | 2.00 |

**Attainment Levels:** 2, 3, 4, 4, 3, 3

**CO Attainment levels:**
- Weights for attainment levels: L1 - Easy-0.75, L2 - Comfortable-0.7, L3 – Medium – 0.65, L4 – Somewhat difficult – 0.6, L5 – Difficult – 0.55
- CO1 – L2, CO2 – L3, CO3 – L4, CO4 – L4, CO5 – L3 and CO6 – L3

**Future Course Mapping:** Advanced Deep Learning, Natural Language Processing, Computer Vision, Generative AI, Reinforcement Learning

**Job Mapping:** Deep Learning Engineer, AI Engineer, Computer Vision Engineer, Generative AI Engineer, Robotics AI Engineering.

---
