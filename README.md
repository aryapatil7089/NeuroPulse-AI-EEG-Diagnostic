# 🧠 NeuroPulse AI: Real-Time EEG Seizure Triage Portal

**Author:** Arya Patil  
**Institution:** Imarticus Learning (Postgraduate Program in Data Science & Analytics)  
**Track:** Data Science Project Competition (July '26)

---

### 🚀 Live Web Application
**[👉 CLICK HERE TO ACCESS THE LIVE DIAGNOSTIC PORTAL](https://neuropulse-ai-msa3.onrender.com)**

---

### 📺 Live System Demonstration
**[👉 CLICK HERE TO WATCH THE CLOUD DEPLOYMENT & UI DEMO](https://drive.google.com/drive/folders/1sTB1kVE0n4wdj-C2vFWHUFTHin9S3gyr?usp=sharing)**
* **Video Verification Data:** The live demonstration utilizes real-world unseen patient telemetry sourced directly from the **[University of Bonn Epileptology Database](https://www.upf.edu/web/ntsa/downloads/-/asset_publisher/xvT6E4pczrBw/content/2001-indications-of-nonlinear-deterministic-and-finite-dimensional-structures-in-time-series-of-brain-electrical-activity-dependence-on-recording-regi)** to validate the model's clinical efficacy.

---

## 🏥 Problem Statement
In intensive care units, continuous EEG monitoring generates massive, multi-hour files containing millions of data points. Neurologists currently suffer from severe diagnostic fatigue by manually scrolling through these files to locate brief seizure events. This creates a dangerous bottleneck, increasing the risk of human error and delayed emergency triage.

## 🚀 Overall Solution Approach
NeuroPulse AI is an end-to-end cloud diagnostic portal designed to automate the first-pass triage of EEG data.
* **Clinical Dataset:** Utilized the gold-standard 178 Hz continuous EEG dataset from the Department of Epileptology at the University of Bonn.
* **Strategic Binarization:** Engineered the original 5-class target into a binary clinical focus (Seizure vs. Non-Seizure).
* **Handling Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to correct the resulting 80:20 class imbalance, establishing a 50:50 ratio to prevent majority-class bias.
* **Cloud Architecture:** Deployed a Python Flask backend on a scalable cloud server architecture featuring Batch Processing to handle multi-hour file uploads without memory overflow.

## 📊 Key Findings
In medical diagnostics, maximizing **Recall (Sensitivity)** is the highest priority to prevent catastrophic False Negatives (missing a seizure).
* **The Winning Model:** The **Base Random Forest** algorithm trained on **SMOTE-balanced data** achieved the highest clinical performance, delivering a **Recall of ~98%**.
* **Rejecting Hyperparameter Trade-offs:** While hyperparameter tuning generally improves overall accuracy, we found it forced a Precision-Recall trade-off. For example, tuning the SVM model decreased its Recall to boost Precision—the exact opposite of our clinical needs. We actively chose the Base Random Forest with SMOTE as it maximized our ability to catch seizures, missing only 13 instances across thousands of test samples. 
* **Why Not SVM?** Furthermore, Support Vector Classifiers were rejected due to their high time and space complexity, making them unsuitable for our lightweight cloud deployment.
* **Baseline Failure:** Naive models (like Logistic Regression) achieved high accuracy but failed clinically, missing >90% of seizures due to the severe class imbalance.

## 💻 Clinical Diagnostic UI (Frontend)
The web application provides instant triage via:
1. **Critical Alert Engine:** Instantly flags files containing seizure activity.
2. **Temporal Diagnostic Map:** Translates multi-hour arrays into a color-coded horizontal timeline (Red = Seizure, Green = Normal).
3. **AJAX Waveform Scrubber:** Utilizes headless background rendering (`Matplotlib Agg`) to allow neurologists to seamlessly scroll through the timeline and visually verify raw EEG waveforms in real-time without page reloads.

## 📁 Repository Structure
* [`app.py`](./app.py): The Flask cloud server backend.
* [`NeuroPulse_Model.ipynb`](./NeuroPulse_Model.ipynb): Complete EDA, preprocessing, and model training notebook.
* [`seizure_scaler.pkl`](./seizure_scaler.pkl): The serialized standard scaler object.
* [`seizure_rf_model.pkl`](./seizure_rf_model.pkl): containing a pre-trained model serialized using Python's pickle module for quick deployment and inference.
* [`templates/`](./templates): Folder containing HTML interface files (`index.html`).
* [`requirements.txt`](./requirements.txt): Required dependencies for cloud deployment.
* [`esr.csv`](./esr.csv): The primary dataset file used for model training and evaluation.
* [`NeuroPulse_AI_Presentation.pptx`](./NeuroPulse_AI_Presentation.pptx): Project presentation slides.

## 🔮 Future Scope
* Upgrade the ingestion pipeline to support live WebSocket streaming directly from hospital IoT EEG caps.
* Scale the AWS infrastructure using a Gunicorn/Nginx cluster for multi-hospital enterprise deployment.

## ⚙️ How to Run the Project Locally

1. Clone this repository to your local machine.
2. Place the downloaded `rf_model.pkl` in the root directory (in the same folder as `app.py`).
3. Install dependencies: `pip install -r requirements.txt`
4. Run the server: `python app.py`
