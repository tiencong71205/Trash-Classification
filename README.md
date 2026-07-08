# ♻️ Trash Classification

A deep learning project for automatic classification of household waste into three recyclable categories using image classification techniques. This project aims to support intelligent waste sorting systems and promote efficient recycling.

---

# 📖 Introduction

The **Trash Classification** project aims to build a deep learning-based system for automatically classifying household waste from images.

The dataset includes **three main categories of recyclable waste**:

- 📄 **Paper** — Paper, newspapers, cardboard, cartons, etc.
- 🥤 **Plastic** — Plastic bottles, plastic bags, plastic containers, etc.
- 🍾 **Glass** — Glass bottles, cups, jars, and other glass products.

The objective of this project is to support automatic waste sorting, improve recycling efficiency, and contribute to environmental protection.

Potential applications include:

- 📷 Camera-based waste classification systems
- 🤖 Smart waste sorting and recycling robots
- 🎓 Research and educational projects in Computer Vision and Environmental AI

---

# 📂 Data Source

The dataset used in this project is publicly available on Kaggle.

**Dataset:**
https://drive.google.com/drive/folders/1abCFClLylTc1pPkuFAUIJNnzC3zEIuWi?usp=drive_link

The dataset contains approximately **15,000 images** collected from common household waste in Vietnam and is organized into three classes:

- `paper`
- `plastic`
- `glass`

The dataset is divided into training, validation, and testing sets for model development and evaluation.

---

# 📁 Dataset Structure

```
dataset/
├── train/
│   ├── paper/
│   ├── plastic/
│   └── glass/
│
├── val/
│   ├── paper/
│   ├── plastic/
│   └── glass/
│
└── test/
    ├── paper/
    ├── plastic/
    └── glass/
```

---

# 🛠 Technologies

- Python 3.10
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow

---

# 🚀 Installation

Clone this repository:

```bash
git clone https://github.com/your-username/trash-classification.git
cd trash-classification
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Training

Train the classification model:

```bash
python train.py
```

You can modify training parameters such as:

- Number of epochs
- Batch size
- Learning rate
- Image size

according to your requirements.

---

# 🔍 Evaluation

Evaluate the trained model:

```bash
python test.py
```

---

# 🖼 Prediction

Predict a single image:

```bash
python predict.py --image path/to/image.jpg
```

---

# 📊 Evaluation Metrics

The model performance is evaluated using:

- Accuracy
- Training Loss
- Validation Loss

---

# 📂 Project Structure

```
Trash-Classification/
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
├── checkpoints/
├── results/
│
├── train.py
├── test.py
├── predict.py
├── requirements.txt
└── README.md
```

---

# 🎯 Future Improvements

- Increase the dataset size.
- Apply advanced data augmentation techniques.
- Compare different deep learning architectures.
- Improve classification accuracy.
- Deploy the model on edge devices for real-time waste classification.

---

# 👨‍💻 Author

**Nguyen Cong**

Faculty of Information Technology

---

# 📄 License

This project is developed for educational and research purposes.