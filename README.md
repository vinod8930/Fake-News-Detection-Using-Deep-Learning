📰 Fake News Detection Using Bi-LSTM
📘 Overview

This project implements a Deep Learning–based Fake News Detection system using a Bidirectional Long Short-Term Memory (Bi-LSTM) network.
The model is designed to identify whether a given news article is real or fake based solely on its textual content.

With the rapid spread of misinformation on social media and online platforms, this project aims to automate fake news detection using modern Natural Language Processing (NLP) and Deep Learning techniques.

🎯 Objectives

Develop a deep learning model (Bi-LSTM) to detect fake news efficiently.

Process and analyze textual data using tokenization, padding, and embeddings.

Achieve high accuracy and reliability across multiple datasets.

Support deployment in real-time environments through APIs or web applications.

🧠 Methodology

Dataset Preparation

Data collected from open-source fake news repositories (train.csv, test.csv).

Text preprocessing: tokenization, lowercasing, stopword removal, and padding.

Model Architecture

Embedding Layer: Converts words into dense vectors.

Bi-LSTM Layers: Capture forward and backward dependencies for better context understanding.

Dropout Layers: Reduce overfitting.

Dense Layer: Classifies news as Fake (1) or Real (0) using sigmoid activation.

Training

Optimizer: Adam

Loss Function: Binary Crossentropy

Evaluation Metrics: Accuracy, Precision, Recall, F1-score

Achieved Accuracy: ≈ 92%

Output

The model generates predictions and saves them to submit.csv.

🧩 Technologies Used
Category	Tools / Libraries
Programming Language	Python 3.x
Deep Learning	TensorFlow, Keras
Data Processing	NumPy, Pandas
NLP Tools	Keras Tokenizer, pad_sequences
Visualization	Matplotlib, Seaborn
Model Deployment	Flask / Streamlit (Future Scope)
⚙️ Hardware & Software Requirements

CPU: Intel i7/Ryzen 7 or higher

GPU: NVIDIA RTX (for faster training)

RAM: 16GB or more

Software:

Python 3.10+

TensorFlow 2.x

scikit-learn, pandas, numpy

🚀 How to Run
# 1. Clone the repository
git clone https://github.com/<your-username>/Fake-News-Detection.git
cd Fake-News-Detection

# 2. Create a virtual environment
conda create -n fake-news python=3.10
conda activate fake-news

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the model
python main.py

📈 Results

Training Accuracy: ~92%

Validation Accuracy: ~90–91%

The Bi-LSTM model successfully captured linguistic cues and outperformed traditional ML methods.

🔮 Future Work

Integrate Transformer-based models (BERT, RoBERTa) for deeper semantic understanding.

Extend to multilingual fake news detection.

Develop a real-time web application for instant news verification.

Implement Explainable AI (XAI) for better interpretability.

👨‍💻 Authors

P Vinod (22BCE8930)

P Bhuvan Sri Satya (22BCE8647)

M Mihira Datta (22BCE7636)

V Sarath (22BCE9625)

Under the guidance of
Dr. K. G. Suma
Associate Professor, SCOPE, VIT-AP

📚 References

IEEE research papers on fake news detection using LSTM, Bi-LSTM, and Transformer architectures (2022–2025).

Public fake news datasets such as LIAR, ISOT, and FakeNewsNet.
