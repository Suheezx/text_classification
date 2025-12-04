# 📰 News Category Classifier  
**Sklearn + TF-IDF + Naive Bayes / Logistic Regression / SVM**  
**With Flask Web Interface**

---

## 📌 1. Project Description  
Энэхүү төсөл нь англи хэл дээрх мэдээний өгөгдлийг ашиглан **текст ангилах (news classification)** машин сургалтын систем юм.  
TF-IDF векторчлол + машин сургалтын 3 өөр алгоритмыг (NB, LR, SVM) туршиж, аль нь хамгийн сайн гүйцэтгэлтэйг харьцуулсан.

Төсөлд дараах боломжууд багтсан:  
✔ Text preprocessing (cleaning, lemmatization, stopwords removal)  
✔ TF-IDF vectorization  
✔ Naive Bayes / Logistic Regression / Linear SVM загварууд  
✔ Confusion matrix график  
✔ Сургасан моделийг `.joblib` хэлбэрээр хадгалах  
✔ Flask веб интерфейсээр текстийг ангилж харах  
✔ Prediction history хадгалах боломж  

---

## 📌 2. Dataset Information

Source: HuffPost News Category Dataset
Dataset нь нийт ~200,000 мэдээний өгөгдөлтэй:  
- title  
- short_description  
- category  

зэрэг багануудтай JSON Lines форматтай.

Dataset эх сурвалж:  
- Kaggle (HuffPost News Category)   

### ✔ Data Preparation  
- `title` + `short_description` → нэг текст болгон нэгтгэсэн  
- Текстийг жижиг үсэг болгох  
- Цэвэрлэх (`[^a-z\s]`)  
- Stopwords устгах (NLTK)  
- Lemmatization (WordNet)  
- Category balancing: нэг ангиллаас **1000 мөр өгөгдөл ширхэг** сонгосон (`max_per_category=1000`)
---
## 📌 1. Project Description  

Төсөлд дараах боломжууд багтсан:  
- text_classification
       - ажиллаж дууссаны дараа confusion matrix гарч ирнэ уг процессыг хаан дараагын үйлдлүүдыг хийнэ.
- preprocess.py
-  app.py

---
## 📌 4. Installation
```bash
git clone https://github.com/Suheezx/text_classification.git
cd project-folder
