import nltk
import pdfplumber
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import os

# suppress undefined metrics warnings
warnings.simplefilter("ignore", UndefinedMetricWarning)

# download nltk resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('English'))

# function to extract text from pdf using pdfplumber
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text

# function to preprocess text: tokenize and lemmatize (ignore stopwords and non-alphabetical words)
def preprocess_text(text):
    # tokenize
    tokens = word_tokenize(text)

    # if word is alphabetical and not a stopword, lemmatize
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha() and word not in stop_words]

    return " ".join(tokens)

# path to resumes
paths = [
    "/Users/tiffanyho/Desktop/CS 6320 Project/Training Resumes/Software_Developer_Resume_1.pdf",
    "/Users/tiffanyho/Desktop/CS 6320 Project/Training Resumes/Software_Developer_Resume_2.pdf",
    "/Users/tiffanyho/Desktop/CS 6320 Project/Training Resumes/Software_Developer_Resume_3.pdf",
    "/Users/tiffanyho/Desktop/CS 6320 Project/Training Resumes/CPA_Resume_1.pdf",
    "/Users/tiffanyho/Desktop/CS 6320 Project/Training Resumes/CPA_Resume_2.pdf",
    "/Users/tiffanyho/Desktop/CS 6320 Project/Training Resumes/CPA_Resume_3.pdf",
    "/Users/tiffanyho/Desktop/CS 6320 Project/Training Resumes/Chemical_Engineer_Resume.pdf",
    "/Users/tiffanyho/Desktop/CS 6320 Project/Training Resumes/Electrical_Engineer_Resume.pdf",
    "/Users/tiffanyho/Desktop/CS 6320 Project/Training Resumes/Mechanical_Engineer_Resume.pdf",
    "/Users/tiffanyho/Desktop/CS 6320 Project/Training Resumes/RN_Resume_1.pdf",
    "/Users/tiffanyho/Desktop/CS 6320 Project/Training Resumes/RN_Resume_2.pdf",
    "/Users/tiffanyho/Desktop/CS 6320 Project/Training Resumes/RN_Resume_3.pdf"
]

# read resumes from pdfs
resumes = [extract_text_from_pdf(path) for path in paths]

# labels for corresponding resumes
labels = ["Software Developer", "Software Developer", "Software Developer",
          "Certified Public Accountant (CPA)", "Certified Public Accountant (CPA)", "Certified Public Accountant (CPA)",
          "Engineer", "Engineer", "Engineer",
          "Registered Nurse (RN)", "Registered Nurse (RN)", "Registered Nurse (RN)"
          ]

# preprocess resumes
preprocessed_resumes = [preprocess_text(resume) for resume in resumes]

# convert resumes into a bag-of-words model (numerical form)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_resumes)

# convert labels to a pandas series
y = pd.Series(labels)

# split resumes into training and testing sets
# test_size=0.3 -> 30% for testing, 70% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# initialize and train Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_train, y_train)

# make predictions on test resumes
y_pred = nb.predict(X_test)

# evaluate model
print("Model Evaluation on Training Resumes: ")
print(classification_report(y_test, y_pred))

# function to predict job role of a new resume
def predict_job_role_from_resume(resume_path):
    # extract and preprocess resume text from pdf
    resume_text = extract_text_from_pdf(resume_path)

    preprocessed_text = preprocess_text(resume_text)

    # vectorize input resume
    vectorized_input = vectorizer.transform([preprocessed_text])

    # predict job role
    predicted_label = nb.predict(vectorized_input)
    return predicted_label[0]

# function to predict job role for multiple resumes in a folder
def predict_job_roles(folder_path):
    resume_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

    predictions = {}
    for resume_file in resume_files:
        resume_path = os.path.join(folder_path, resume_file)
        predicted_role = predict_job_role_from_resume(resume_path)
        predictions[resume_file] = predicted_role

    return predictions

# path to test resumes folder
test_folder_path = "/Users/tiffanyho/Desktop/CS 6320 Project/Test Resumes"

# predict job roles for test resumes
predictions = predict_job_roles(test_folder_path)

# print predicted job roles
for resume_name, role in predictions.items():
    print(f"Resume: {resume_name}, Predicted Job Role: {role}")
