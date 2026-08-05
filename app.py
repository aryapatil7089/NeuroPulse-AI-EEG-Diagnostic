from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg') # Required for server background plotting
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load your Machine Learning Model and Scaler
try:
    model = pickle.load(open('seizure_rf_model.pkl', 'rb'))
    scaler = pickle.load(open('seizure_scaler.pkl', 'rb'))
except Exception as e:
    print(f"Warning: Model or scaler not found. Please upload them! Error: {e}")

# Global variables to hold state for the scrubber
LATEST_ROWS_LIST = []
LATEST_PREDICTIONS = []
LATEST_HZ = 178

def generate_waveform_plot(data_row, hz, current_sec, is_seizure):
    """Generates the raw EEG wave graph. Red for seizure, Green for normal."""
    plt.figure(figsize=(10, 3), facecolor='#0b0f19')
    ax = plt.axes()
    ax.set_facecolor('#0b0f19')
    
    time_axis = np.linspace(0, 1, len(data_row))
    
    # Dynamic Color: Red if Seizure, Green if Normal
    line_color = '#ef4444' if is_seizure else '#10b981'
    plt.plot(time_axis, data_row, color=line_color, linewidth=1.5)
    
    status_text = "SEIZURE ACTIVITY DETECTED" if is_seizure else "NORMAL BRAINWAVE"
    plt.title(f'Raw EEG Waveform - Second {current_sec} ({status_text})', color=line_color, pad=10, weight='bold')
    plt.xlabel('Time (Seconds)', color='#9ca3af')
    plt.ylabel('Amplitude (µV)', color='#9ca3af')
    plt.tick_params(colors='#9ca3af')
    
    # Clean borders
    ax.spines['bottom'].set_color('#1f2937')
    ax.spines['top'].set_color('#1f2937') 
    ax.spines['right'].set_color('#1f2937')
    ax.spines['left'].set_color('#1f2937')
    plt.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def generate_timeline_plot(predictions):
    """Generates the Green/Red temporal timeline map"""
    plt.figure(figsize=(10, 1.5), facecolor='#0b0f19')
    ax = plt.axes()
    ax.set_facecolor('#0b0f19')
    
    colors = ['#ef4444' if p == 1 else '#10b981' for p in predictions]
    
    # Added edge colors so blocks are easily countable
    plt.bar(range(1, len(predictions) + 1), [1]*len(predictions), color=colors, width=1.0, edgecolor='#0b0f19', linewidth=1.0)
    
    # Force the X-axis to show every individual second (1, 2, 3...) instead of skipping to 5
    step = 1 if len(predictions) <= 30 else 2
    plt.xticks(range(1, len(predictions) + 1, step))
    
    plt.yticks([])
    plt.xlabel('Time (Seconds)', color='#9ca3af')
    plt.tick_params(axis='x', colors='#9ca3af', labelsize=9)
    
    ax.spines['bottom'].set_color('#1f2937')
    ax.spines['top'].set_color('none') 
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global LATEST_ROWS_LIST, LATEST_HZ, LATEST_PREDICTIONS
    try:
        preset_hz = request.form.get('sampling_rate_preset', '178')
        user_hz = float(request.form.get('custom_hz', 178)) if preset_hz == 'custom' else float(preset_hz)
        LATEST_HZ = user_hz

        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            
            # SMART MEMORY CAP: Only read the first 20,000 rows to prevent RAM crash on 1GB servers
            df = pd.read_csv(file, header=None, sep=r'\s+|,', engine='python', nrows=20000)
            raw_values = df.values.astype(float)
            
            if raw_values.shape[1] == 1 or len(raw_values.shape) == 1:
                flat_stream = raw_values.flatten()
                
                # RAPID TRIAGE FEATURE: Cap the analysis at exactly 60 seconds
                max_points = int(user_hz * 60)
                if len(flat_stream) > max_points:
                    flat_stream = flat_stream[:max_points]
                    
                total_seconds = max(1, round(len(flat_stream) / user_hz))
                raw_rows_list = np.array_split(flat_stream, total_seconds)
            else:
                # RAPID TRIAGE FEATURE: Cap at 60 rows (60 seconds) if data is matrix
                if len(raw_values) > 60:
                    raw_values = raw_values[:60]
                raw_rows_list = raw_values
        else:
            input_data = request.form.get('eeg_values', '').strip()
            
            # PREVENT BLANK SUBMISSION CRASH
            if not input_data:
                return render_template('index.html', prediction_text="⚠️ Please provide data! Upload an EEG file or paste a raw waveform array to begin.", status="danger")
                
            features = [float(x.strip()) for x in input_data.split(',') if x.strip() != '']
            raw_rows_list = [np.array(features)]

        LATEST_ROWS_LIST = raw_rows_list
        processed_rows = []
        
        # Format the data exactly how the ML model expects it (178 features)
        for row in raw_rows_list:
            if len(row) < 178:
                row = np.pad(row, (0, 178 - len(row)), 'constant')
            elif len(row) > 178:
                row = row[:178]
            processed_rows.append(row)

        processed_rows = np.array(processed_rows)
        scaled_features = scaler.transform(processed_rows)
        all_predictions = model.predict(scaled_features)
        
        # Save predictions globally so the scrubber knows what color to make the graph
        LATEST_PREDICTIONS = all_predictions
        
        seizure_detected = 1 in all_predictions
        total_seizure_seconds = sum(all_predictions)
        
        # Render the First Second Graph by default
        target_idx = 0
        is_seizure_sec = all_predictions[target_idx] == 1
        waveform_url = generate_waveform_plot(raw_rows_list[target_idx], user_hz, target_idx + 1, is_seizure_sec)
        timeline_url = generate_timeline_plot(all_predictions) if len(all_predictions) > 1 else None
        
        if seizure_detected:
            result = f"⚠️ CRITICAL ALERT: Epileptic Seizure Activity Detected! (Flagged in {total_seizure_seconds} out of {len(all_predictions)} seconds analyzed via Rapid Triage)"
            status_class = "danger"
        else:
            result = f"✅ Patient Stable: Normal Brain Baseline Rhythm verified across the first {len(all_predictions)} seconds via Rapid Triage."
            status_class = "success"
            
        return render_template('index.html', prediction_text=result, waveform_url=waveform_url, timeline_url=timeline_url, status=status_class, total_secs=len(all_predictions), current_sec=target_idx+1)
        
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error processing data: {str(e)}", status="danger")

# AJAX Route: Triggers when user drags the slider
@app.route('/scrub', methods=['POST'])
def scrub():
    global LATEST_ROWS_LIST, LATEST_HZ, LATEST_PREDICTIONS
    try:
        target_sec = int(request.form.get('target_second', 1))
        target_idx = target_sec - 1
        
        if target_idx < 0 or target_idx >= len(LATEST_ROWS_LIST):
            return jsonify({'success': False, 'error': 'Invalid second'})
            
        is_seizure_sec = LATEST_PREDICTIONS[target_idx] == 1
        new_image_b64 = generate_waveform_plot(LATEST_ROWS_LIST[target_idx], LATEST_HZ, target_sec, is_seizure_sec)
        
        return jsonify({'success': True, 'new_image': new_image_b64})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)