# PhishScan: Automated Phishing Detection with Machine Learning 🕵️‍♂️🌐

---

## Project Authors
- **Jack Hsieh** (yhsieh37@asu.edu)
- **Siddharth Ranjan** (sranja18@asu.edu)
- **Ian Oxley** (ioxley@asu.edu)
- **Naveen Kumar** (nmanokar@asu.edu)
- **Willem Grier** (wgrier@asu.edu)

---

## 📌 Abstract
Phishing attacks have become a prominent cyber threat, exploiting vulnerabilities in individuals and businesses alike. **PhishScan** leverages machine learning models to detect phishing websites by analyzing URL structures, syntax, web traffic, and content metadata. By embedding URLs into models and comparing their impact across classifiers, this project demonstrates the efficacy of machine learning in cybersecurity.

Key features include:
- Comparison of baseline models like Logistic Regression, SVM, Decision Trees, and advanced models such as XGBoost and Graph Neural Networks.
- Integration of BERT-based URL embeddings for enhanced text feature understanding.
- Analysis of feature selection methods like Boruta to streamline computational efficiency.

---

## ✨ Motivation
Phishing attacks are among the most effective forms of cybercrime, with financial institutions and reputable corporations being prime targets. Despite efforts to educate individuals on phishing detection, advancements in AI-generated phishing techniques (e.g., WormGPT, FraudGPT) have outpaced human learning capabilities. PhishScan proposes an automated, AI-driven solution to classify phishing and legitimate websites accurately.

### Key Challenges:
1. The rapid improvement of phishing tactics.
2. Difficulty in differentiating AI-generated phishing content.
3. The need for efficient feature extraction and classification in time-critical scenarios.

---

## 📂 Dataset
The **Web Page Phishing Detection Dataset** [3] contains:
- **11430 websites** (balanced with 50% phishing sites).
- **88 features** extracted from URL structure, syntax, web page content, and web traffic metadata.
- Dataset Link: [Web Page Phishing Detection](https://archive.ics.uci.edu/ml/datasets/phishing+websites)

---

## 📚 Technical Background

### Classifiers Used:
1. **Logistic Regression**
2. **Support Vector Machines (SVM)**
3. **Decision Tree**
4. **Recurrent Neural Network (RNN)**
5. **Graph Neural Network (GNN)**
6. **Random Forest**
7. **XGBoost**
8. **K-Nearest Neighbors (KNN)**

### Feature Selection:
- **Boruta**: Identifies significant features to reduce model complexity and computational overhead.

---

## 🚀 Methodology

### Steps:
1. **Data Preprocessing**:
   - Cleaned and normalized features using Min-Max Scaling.
   - Tokenized URLs using BERT-Tiny for embeddings.
2. **Feature Selection**:
   - Evaluated models with and without Boruta feature selection.
3. **Model Training**:
   - Implemented baseline models and advanced architectures with 5-fold cross-validation.
   - Fine-tuned hyperparameters for optimal performance.
4. **Evaluation**:
   - Metrics: F1 Score, Accuracy, Sensitivity, Specificity.
   - Compared models with and without URL embeddings.

---

## 🔬 Experiments

### Scenarios:
1. **All Features Except URL**: Baseline model excluding URL features.
2. **All Features Except URL (With Feature Selection)**: Reduced feature set using Boruta.
3. **All Features Including URL**: Added BERT-generated embeddings for the URL column.

### Model-Specific Insights:
- **Recurrent Neural Network (RNN)**:
  - Incorporates LSTM for sequential feature learning.
  - Achieved an F1 score of **0.9628** when using URL embeddings.
- **Graph Neural Network (GNN)**:
  - Models relationships between feature nodes.
  - F1 score improved to **0.6666** with URL embeddings.
- **Logistic Regression**:
  - Simple yet effective for text-based feature classification.
  - F1 score reached **0.9639** with URL embeddings.
- **XGBoost**:
  - Best-performing classifier overall with an F1 score of **0.9737**.
- **Random Forest**:
  - Robust ensemble model. Showed reduced complexity with Boruta.

---

## 📊 Results

| Model               | No URL (No Boruta) | No URL (Boruta) | URL (No Boruta) | URL (Boruta) |
|---------------------|--------------------|-----------------|-----------------|--------------|
| Logistic Regression | 0.9439 ± 0.003    | 0.9327 ± 0.003  | 0.9639 ± 0.003  | -            |
| SVM                 | 0.9576 ± 0.003    | 0.9536 ± 0.003  | 0.9702 ± 0.003  | -            |
| Decision Tree       | 0.9384 ± 0.003    | 0.9387 ± 0.003  | 0.9402 ± 0.003  | -            |
| RNN                 | 0.9614 ± 0.003    | 0.9621 ± 0.002  | 0.9613 ± 0.004  | 0.9628 ± 0.003 |
| GNN                 | 0.6334 ± 0.003    | -               | 0.6666 ± 0.008  | -            |
| Random Forest       | 0.9638 ± 0.004    | 0.9646 ± 0.008  | 0.9624 ± 0.004  | 0.9621 ± 0.003 |
| XGBoost             | 0.9713 ± 0.003    | 0.9702 ± 0.003  | 0.9737 ± 0.003  | -            |
| KNN                 | 0.9499 ± 0.003    | 0.9383 ± 0.003  | 0.9408 ± 0.003  | -            |

### Key Observations:
1. URL embeddings significantly improved F1 scores across most models.
2. Boruta feature selection reduced complexity but did not consistently enhance performance.
3. XGBoost proved to be the most robust model overall.

---

## 🔮 Conclusion

PhishScan demonstrates the potential of AI-driven phishing detection:
- **URL Embeddings**: Proved to be a critical feature, enhancing performance across most classifiers.
- **Feature Selection**: Useful for simplifying models but less impactful for this balanced dataset.
- **Best Model**: XGBoost achieved the highest F1 score of **0.9737** with URL embeddings.

Future work includes:
- Extending the dataset with real-time phishing URLs.
- Exploring more advanced deep learning models like transformers for end-to-end URL analysis.

---

## 📜 References

1. APWG (2023). Phishing Activity Trends Report, 2nd Quarter 2023.
2. Devlin, J., Chang, M., Lee, K., and Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
3. Hannousse, A. (2021). Web Page Phishing Detection Dataset.
4. Hua, S., Li, X., Jing, Y., and Liu, Q. (2022). A Semantic Hierarchical Graph Neural Network for Text Classification.
5. IC3, F. (2021). Internet Crime Report.
6. Saha, I., et al. (2020). Phishing Attacks Detection Using Deep Learning Approach.
7. SlashNext (2023). The State of Phishing 2023.
8. Tessian (2023). Psychology of Human Error 2022.

---

## Kaggle Notebook and Document Links

https://docs.google.com/spreadsheets/d/1vO7BbM3pM7NZCmql8z2n5EFdh2xLxqN0NS7zprdT-08/edit?gid=0#gid=0
https://www.kaggle.com/code/wgrier/cse575-randomforestclassifier
https://colab.research.google.com/drive/1EtypK9Q8mnICjpt19BXIgw1S3cOohH-X?usp=sharing
https://www.kaggle.com/code/ianoxley/cse575-rnn-url
https://colab.research.google.com/drive/1RFHN8k9gRP6FZcHrS2hGY8J66PU81ouC?usp=sharing
https://www.kaggle.com/code/naveenkumarmanokaran/phishing-dataset-analysis?scriptVersionId=165989833

