# Pneumonia Detection Using X-Ray Images

This project leverages deep learning to classify chest X-ray images and detect pneumonia automatically. Using convolutional neural networks (CNNs), it aims to assist medical professionals by providing fast and reliable predictions from radiographic images.

---

## 📁 Project Structure

- `XRAY_CLASSIFICATION.ipynb` — Main Jupyter notebook containing data preprocessing, model training, evaluation, and visualization.
- `README.md` — Project documentation.

---

## 🧠 Model Overview

The model uses a deep CNN architecture (EfficientNet) for image classification. The workflow includes:

1. **Dataset Loading & Preprocessing**  
   - Chest X-ray images are resized, normalized, and split into training, validation, and testing sets.

2. **Model Building**  
   - A CNN is fine-tuned on the dataset to distinguish between *Normal* and *Pneumonia* cases.

3. **Training**  
   - Trained using transfer learning with early stopping and checkpoints to prevent overfitting.

4. **Evaluation**  
   - Model accuracy, confusion matrix, and classification report are generated to assess performance.

---

## 📊 Results

- High classification accuracy achieved on the test set.
- Clear visualization of correctly and incorrectly classified samples.
- Confusion matrix and metrics provide insights into model reliability.

---

## 🧪 Requirements

Install the following Python dependencies:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn
```
For GPU acceleration:
```bash
pip install tensorflow-gpu
```
▶️ How to Run
### Clone the repository
```bash
git clone https://github.com/your-username/xray-classification.git
cd xray-classification
```
### Install dependencies
```bash
pip install -r requirements.txt
```
### Place the dataset

Download and extract the chest X-ray dataset into a data/ folder.

### Run the notebook

jupyter notebook XRAY_CLASSIFICATION.ipynb

Save the trained model for later deployment or inference.

## 🌐 Deployment
This model can be deployed with a simple Streamlit interface to allow users to upload X-ray images and receive instant predictions:

streamlit run app.py
## 📄 License
This project is open-source and available under the MIT License.

## ✍️ Author
Developed by Netsanet Teklegiorgis Brhane — as part of a deep learning project focused on medical imaging.

## 💡 Acknowledgements

Dataset: NIH Chest X-ray Dataset

TensorFlow & Keras for model building

Matplotlib & scikit-learn for visualization and evaluation
