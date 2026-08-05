# 🧠 NeuroPulse AI: Real-Time EEG Seizure Triage Portal

**Author:** Arya Patil  
**Institution:** Imarticus Learning (Postgraduate Program in Data Science & Analytics)  
**Track:** Data Science Project Competition (July '26)

---

### 📺 Live System Demonstration
**[👉 CLICK HERE TO WATCH THE CLOUD DEPLOYMENT & UI DEMO](https://drive.google.com/drive/folders/1sTB1kVE0n4wdj-C2vFWHUFTHin9S3gyr?usp=sharing)**

---

## 🏥 Problem Statement
In intensive care units, continuous EEG monitoring generates massive, multi-hour files containing millions of data points. Neurologists currently suffer from severe diagnostic fatigue by manually scrolling through these files to locate brief seizure events. This creates a dangerous bottleneck, increasing the risk of human error and delayed emergency triage.

## 🚀 Overall Solution Approach
NeuroPulse AI is an end-to-end cloud diagnostic portal designed to automate the first-pass triage of EEG data.
* **Clinical Dataset:** Utilized the gold-standard 178 Hz continuous EEG dataset from the Department of Epileptology at the University of Bonn.
* **Strategic Binarization:** Engineered the original 5-class target into a binary clinical focus (Seizure vs. Non-Seizure).
* **Handling Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to correct the resulting 80:20 class imbalance, establishing a 50:50 ratio to prevent majority-class bias.
* **Cloud Architecture:** Deployed a Python Flask backend on an **AWS EC2 (t3.micro)** server featuring Scalable Batch Processing to handle multi-hour file uploads without memory overflow.

## 📊 Key Findings
* **Baseline Failure:** Naive models (like Logistic Regression) achieved high accuracy but failed clinically, missing >90% of seizures due to the severe class imbalance.
* **The Winning Model:** Post-SMOTE, the hyperparameter-tuned **Random Forest** ensemble outperformed LightGBM and SVMs by prioritizing recall. 
* **Clinical Safety:** The final model achieved **>97% Recall**, minimizing False Negatives (missing only 13 seizure instances across thousands of unseen test samples).

## 💻 Clinical Diagnostic UI (Frontend)
The web application provides instant triage via:
1. **Critical Alert Engine:** Instantly flags files containing seizure activity.
2. **Temporal Diagnostic Map:** Translates multi-hour arrays into a color-coded horizontal timeline (Red = Seizure, Green = Normal).
3. **AJAX Waveform Scrubber:** Utilizes headless background rendering (`Matplotlib Agg`) to allow neurologists to seamlessly scroll through the timeline and visually verify raw EEG waveforms in real-time without page reloads.

## 📁 Repository Structure
* `app.py`: The Flask cloud server backend.
* `NeuroPulse_Model.ipynb`: Complete EDA, preprocessing, and model training notebook.
* `rf_model.pkl` & `scaler.pkl`: The serialized Random Forest model and standard scaler.
* `templates/`: HTML interface files.
* `requirements.txt`: Required dependencies for cloud deployment.

## 🔮 Future Scope
* Upgrade the ingestion pipeline to support live WebSocket streaming directly from hospital IoT EEG caps.
* Scale the AWS infrastructure using a Gunicorn/Nginx cluster for multi-hospital enterprise deployment.

## ⚙️ How to Run the Project Locally
Due to GitHub's 25MB web upload limit, the trained Random Forest model (45MB) is hosted securely on Google Drive.

1. Clone this repository to your local machine.
2. **[👉 Click Here to Download rf_model.pkl](https://drive.google.com/file/d/1OS2zA2MdjZ91zXNqxrDqQamPu6ZfYFAc/view?usp=sharing)**
3. Place the downloaded `rf_model.pkl` in the root directory (in the same folder as `app.py`).
4. Install dependencies: `pip install -r requirements.txt`
5. Run the server: `python app.py`
