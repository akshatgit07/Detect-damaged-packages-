# Detect-damaged-packages-
Real-Time Image Pipeline for Parcel Damage Detection
This guide outlines a real-time parcel damage detection system using a Kaggle dataset, a Kafka + Spark + TensorFlow pipeline for streaming and training a convolutional neural network (CNN), and a Streamlit dashboard for visualizing predictions.
Combined Objective
Build a system to:

Utilize a Kaggle dataset as the data source.
Implement a Kafka + Spark + TensorFlow pipeline for streaming and training a CNN.
Create a Streamlit dashboard for real-time prediction visualization.

Project Structure & Navigation Guide
The project is organized as follows:
group23/
├── image_producer.py            # Sends image paths and labels to Kafka
├── image_consumer.py            # Spark job that listens to Kafka and writes image metadata to disk
├── train_model_augmented.py     # TensorFlow training script with augmentation and early stopping
├── dashboard.py                 # Streamlit dashboard to visualize model predictions
├── parcel_model_augmented.h5    # Trained model (binary classifier)
├── requirements.txt             # Python dependencies
├── Damaged Package Detection Report.docx
├── damaged-and-intact-packages/ # Image data used for simulation
│   ├── damaged/
│   └── intact/
└── README.md                    # This document


Note: This pipeline was developed and tested on a local terminal.

Kaggle Dataset Info

Source: Damaged and Intact Packages
Description: Dataset containing images of intact and damaged packages.

Setup Instructions
1. Start Kafka Locally
Ensure Zookeeper and Kafka are running on localhost:9092.
# Start Zookeeper (default)
zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka Broker
kafka-server-start.sh config/server.properties

2. Run Kafka Producer
python3 image_producer.py


Important: Update image_dir in image_producer.py to point to your dataset path.

This script streams image paths and labels to the Kafka topic test_2.
3. Run Spark Streaming Consumer
python3 image_consumer.py

This script writes Kafka messages to /tmp/preprocessed_data/ as CSV files.
4. Train TensorFlow Model
python3 train_model_augmented.py


Reads streamed data from /tmp/preprocessed_data/.
Applies data augmentation and class balancing.
Trains a CNN model for binary classification.
Saves the trained model as parcel_model_augmented.h5.

5. Launch Streamlit Dashboard
python3 -m streamlit run dashboard.py

Open your browser at: http://localhost:8501
Dashboard Features

Displays predictions on live or static image folders.
Shows predicted label and confidence score for each image.
Supports drag-and-drop for prediction-ready analysis.

Requirements
Install Python dependencies:
python3 -m pip install -r requirements.txt

Required libraries:

tensorflow
pandas
numpy
streamlit
scikit-learn
kafka-python
pillow

Notes

All scripts assume Kafka is running on localhost:9092.
Ensure image_dir in image_producer.py points to your dataset path.
Verify that /tmp/preprocessed_data is writable.
