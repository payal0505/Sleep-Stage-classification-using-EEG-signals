

import os, glob
import streamlit as st
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import joblib  


st.set_page_config(page_title="Sleep Stage Classification using EEG signals", layout="wide")
st.title("Sleep Stage Classification Dashboard using EEG signals")

#path
basepath = r"D:\PAYAL\SEM 5\DSP\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette"
psgfiles = [f for f in os.listdir(basepath) if f.endswith("-PSG.edf")]
stagenames = ["W", "N1", "N2", "N3", "REM"]
storedresults = {"ML": {}, "DL": {}, "Hybrid": {}}

DEVICE = torch.device("cpu")  

# feature extraction 
def hjorthparameter(sig):
    firstderiv = np.diff(sig)
    secondderiv = np.diff(firstderiv)
    var0 = np.var(sig)
    var1 = np.var(firstderiv)
    var2 = np.var(secondderiv)
    activity = var0
    mobility = np.sqrt(var1 / var0) if var0 != 0 else 0
    complexity = (np.sqrt(var2 / var1) / mobility) if var1 != 0 and mobility != 0 else 0
    return activity, mobility, complexity

def petrosianparameter(sig):
    diff = np.diff(sig)
    if len(diff) < 2: return 0.0
    N_delta = np.sum(diff[:-1] * diff[1:] < 0)
    n = len(sig)
    denom = (np.log10(n) + np.log10(n / (n + 0.4 * N_delta))) if (n + 0.4 * N_delta) != 0 else 1.0
    return np.log10(n) / denom

def hurstexponent(ts):
    if len(ts) < 20: return 0.0
    lags = range(2, min(20, len(ts)//2))
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    if np.any(np.array(tau) <= 0): return 0.0
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def extractfeatures(X_raw):
    X_features = []
    for sig in X_raw:
        psd, _ = mne.time_frequency.psd_array_welch(sig, sfreq=100, fmin=0.5, fmax=40, n_fft=256, verbose=False)
        psd_mean = np.mean(psd) if np.isfinite(psd).all() else 0.0
        hj = hjorthparameter(sig)
        pfd = petrosianparameter(sig)
        h = hurstexponent(sig)
        X_features.append([psd_mean, pfd, *hj, h])
    return np.array(X_features)

# Visualization 
def visualize_predictions(y_true, y_pred, channel_name):
    st.subheader("Predicted vs True Sleep Stages")
    fig_t, ax_t = plt.subplots(figsize=(10,2.8))
    ax_t.plot(y_pred, label="Predicted", marker='o', linestyle='-', markersize=3)
    ax_t.plot(y_true, label="True", marker='x', linestyle='--', alpha=0.6)
    ax_t.set_yticks(range(len(stagenames)))
    ax_t.set_yticklabels(stagenames)
    ax_t.set_xlabel("Epoch")
    ax_t.set_ylabel("Stage")
    ax_t.set_title(f"Predicted vs True ({channel_name})")
    ax_t.legend()
    st.pyplot(fig_t)

    # visualization using pie chart and bar chart
    colA, colB = st.columns(2)
    counts = [int(np.sum(np.array(y_pred)==i)) for i in range(len(stagenames))]

    with colA:
        st.subheader("Bar Chart")
        fig_bar, ax_bar = plt.subplots(figsize=(5,3.5))
        ax_bar.bar(stagenames, counts)
        ax_bar.set_xlabel("Sleep Stages")
        ax_bar.set_ylabel("Epochs")
        st.pyplot(fig_bar)

    with colB:
        st.subheader("Pie Chart")
        fig_pie, ax_pie = plt.subplots(figsize=(5,3.5))
        if sum(counts) == 0:
            ax_pie.text(0.5, 0.5, "No predictions", ha='center')
        else:
            ax_pie.pie(counts, labels=stagenames, autopct="%1.1f%%")
        st.pyplot(fig_pie)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5,3.5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=stagenames, yticklabels=stagenames, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    st.pyplot(fig_cm)

# each Epoch Hypnogram 
def plot_per_epoch_hypnogram(y_true, y_pred, epoch_length=30):
    st.subheader("Time vs Sleep Stages (Per-Epoch Hypnogram)")
    times = np.arange(len(y_true)) * (epoch_length / 3600)  # hours
    bar_width = epoch_length / 3600

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(times, y_true, width=bar_width, align="edge", color="blue", alpha=0.6, label="True")
    ax.bar(times, y_pred, width=bar_width, align="edge", color="red", alpha=0.4, label="Predicted")
    ax.set_yticks(range(len(stagenames)))
    ax.set_yticklabels(stagenames)
    ax.invert_yaxis()
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Sleep Stage")
    ax.set_title("Sleep Stage Progression (Per-Epoch)")
    ax.legend()
    st.pyplot(fig)

# Load EDF and doing preprocessing  
def load_edf(file_path, channel_name):
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        if channel_name not in raw.ch_names:
            st.warning(f"Channel {channel_name} not found in this file.")
            return None, None, None
        raw.pick([channel_name])
        raw.filter(0.5, 40, fir_design="firwin", verbose=False)

        subject_id = os.path.basename(file_path)[:7]
        hyp_files = glob.glob(os.path.join(basepath, f"{subject_id}*Hypnogram.edf"))
        if not hyp_files:
            st.warning("No corresponding Hypnogram file found.")
            return None, None, None

        hyp = mne.read_annotations(hyp_files[0])
        raw.set_annotations(hyp)
        mapping = {
            "Sleep stage W":0,"Sleep stage 1":1,"Sleep stage 2":2,
            "Sleep stage 3":3,"Sleep stage 4":3,"Sleep stage R":4
        }
        events, _ = mne.events_from_annotations(raw, event_id=mapping, verbose=False)
        epochs = mne.Epochs(raw, events, event_id=mapping, tmin=0, tmax=30, baseline=None, preload=True, verbose=False)
        X_raw = epochs.get_data()[:,0,:]
        y = epochs.events[:,-1]
        return X_raw, y, epochs
    except Exception as e:
        st.error(f"Error loading EDF: {e}")
        return None, None, None
    


output_base = r"D:\PAYAL\SEM 5\DSP\processed_data"  

def save_preprocessed_data(subject_id, X_raw, y, epochs, channel_name):
    subject_folder = os.path.join(output_base, subject_id)
    os.makedirs(subject_folder, exist_ok=True)

   
    np.save(os.path.join(subject_folder, f"{channel_name}_X.npy"), X_raw)
    np.save(os.path.join(subject_folder, f"{channel_name}_y.npy"), y)

    joblib.dump(epochs, os.path.join(subject_folder, f"{channel_name}_epochs.pkl"))

    return subject_folder

#  Add Epoch View in Streamlit 
def view_and_save_epochs(X_raw, y, epochs, subject_id, channel_name, key_prefix=""):
    st.subheader("Epoch Viewer")
    epoch_idx = st.number_input(
        "Select epoch index",
        min_value=0,
        max_value=len(X_raw)-1,
        step=1,
        key=f"{key_prefix}_epoch_idx"   
    )

    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(X_raw[epoch_idx])
    ax.set_title(f"Epoch {epoch_idx} - Label: {stagenames[y[epoch_idx]]}")
    st.pyplot(fig)

    if st.button("Save this epoch", key=f"{key_prefix}_save_btn"):
        subject_folder = os.path.join(output_base, subject_id)
        os.makedirs(subject_folder, exist_ok=True)
        np.save(os.path.join(subject_folder, f"{channel_name}_epoch_{epoch_idx}.npy"), X_raw[epoch_idx])
        st.success(f"Epoch {epoch_idx} saved in {subject_folder}")



#  PyTorch Models
class CNNLSTM(nn.Module):
    def __init__(self, input_length, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0,2,1)
        out, (h_n, _) = self.lstm(x)
        x = F.relu(self.fc1(h_n[-1]))
        return self.fc2(x)

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        pool_len = (input_length - 2) // 2
        self.flattened_size = 32 * pool_len
        self.decoder_fc = nn.Linear(self.flattened_size, input_length)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        batch = x.shape[0]
        flat = x.view(batch, -1)
        recon = self.decoder_fc(flat)
        return flat, recon

# training data 
def train_torch_classifier(model, X_train, y_train, epochs=3, batch_size=8, lr=1e-3):
    model.to(DEVICE)
    Xt = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    yt = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
    return model

def train_feature_extractor(fe_model, X_train, epochs=3, batch_size=8, lr=1e-3):
    fe_model.to(DEVICE)
    Xt = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    loader = DataLoader(TensorDataset(Xt, Xt.squeeze(1)), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(fe_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    fe_model.train()
    for _ in range(epochs):
        for Xb, yb in loader:
            optimizer.zero_grad()
            flat, recon = fe_model(Xb)
            loss = criterion(recon, yb)
            loss.backward()
            optimizer.step()
    return fe_model

def predict_torch_model(model, X_input):
    model.to(DEVICE)
    model.eval()
    Xt = torch.tensor(X_input, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        out = model(Xt)
        preds = torch.argmax(out, dim=1).cpu().numpy()
    return preds

tab1, tab2, tab3 = st.tabs(["ML","DL","Hybrid"])

#ml
with tab1:
    st.header("Machine Learning Models")
    selected_file = st.selectbox("Select PSG file (ML)", psgfiles, key="ml_file")
    if selected_file:
        raw = mne.io.read_raw_edf(os.path.join(basepath,selected_file), preload=True, verbose=False)
        channel_name = st.selectbox("Select EEG channel (ML)", raw.ch_names, key="ml_ch")
        if channel_name:
            X_raw, y, epochs = load_edf(os.path.join(basepath,selected_file), channel_name)
            if X_raw is not None and len(X_raw) > 0:
                # saving of preprocessed data
                subject_id = os.path.basename(selected_file)[:7]
                save_preprocessed_data(subject_id, X_raw, y, epochs, channel_name)
                view_and_save_epochs(X_raw, y, epochs, subject_id, channel_name, key_prefix="ML")

                # features extract and model creatin 
                X_features = extractfeatures(X_raw)
                smote = SMOTE(random_state=42)
                try:
                    X_res, y_res = smote.fit_resample(X_features, y)
                except Exception:
                    X_res, y_res = X_features, y
                X_train, X_test, y_train, y_test = train_test_split(
                    X_res, y_res, test_size=0.2, random_state=42,
                    stratify=y_res if len(np.unique(y_res)) > 1 else None
                )
                svm = SVC(kernel="rbf").fit(X_train,y_train)
                rf = RandomForestClassifier().fit(X_train,y_train)
                knn = KNeighborsClassifier().fit(X_train,y_train)

                col1,col2,col3 = st.columns(3)
                col1.metric("SVM Accuracy",f"{accuracy_score(y_test,svm.predict(X_test)):.2%}")
                col2.metric("RF Accuracy",f"{accuracy_score(y_test,rf.predict(X_test)):.2%}")
                col3.metric("KNN Accuracy",f"{accuracy_score(y_test,knn.predict(X_test)):.2%}")

                y_pred = rf.predict(X_features)
                storedresults["ML"][selected_file] = (y_pred,y)
                visualize_predictions(y, y_pred, channel_name)
                plot_per_epoch_hypnogram(np.array(y), np.array(y_pred))
            else:
                st.warning("Could not extract epochs from this file / channel.")


# DL Tab 
with tab2:
    st.header("Deep Learning Model (CNN + LSTM) - PyTorch")
    selected_file = st.selectbox("Select PSG file (DL)", psgfiles, key="dl_file")
    if selected_file:
        raw = mne.io.read_raw_edf(os.path.join(basepath,selected_file), preload=True, verbose=False)
        channel_name = st.selectbox("Select EEG channel (DL)", raw.ch_names, key="dl_ch")
        if channel_name:
            X_raw, y, epochs = load_edf(os.path.join(basepath,selected_file), channel_name)
            if X_raw is not None and len(X_raw) > 0:
                subject_id = os.path.basename(selected_file)[:7]
                save_preprocessed_data(subject_id, X_raw, y, epochs, channel_name)
                view_and_save_epochs(X_raw, y, epochs, subject_id, channel_name, key_prefix="DL")
                
                if selected_file in storedresults["DL"]:
                    y_pred, y_true = storedresults["DL"][selected_file]
                    st.success("Loaded stored predictions")
                    visualize_predictions(y_true, y_pred, channel_name)
                    plot_per_epoch_hypnogram(np.array(y_true), np.array(y_pred))
                else:
                    input_length = X_raw.shape[1]
                    model = CNNLSTM(input_length=input_length, num_classes=5)
                    train_torch_classifier(model, X_raw, y, epochs=3, batch_size=8, lr=1e-3)
                    y_pred_full = predict_torch_model(model, X_raw)
                    acc = accuracy_score(y, y_pred_full)
                    st.metric("DL Accuracy", f"{acc*100:.2f}%")
                    storedresults["DL"][selected_file] = (y_pred_full, y)
                    visualize_predictions(y, y_pred_full, channel_name)
                    plot_per_epoch_hypnogram(np.array(y), np.array(y_pred_full))
            else:
                st.warning("Could not extract epochs from this file / channel.")


# Hybrid
with tab3:
    st.header("Hybrid Model (CNN Feature Extractor + RandomForest)")
    selected_file = st.selectbox("Select PSG file (Hybrid)", psgfiles, key="hyb_file")
    if selected_file:
        raw = mne.io.read_raw_edf(os.path.join(basepath,selected_file), preload=True, verbose=False)
        channel_name = st.selectbox("Select EEG channel (Hybrid)", raw.ch_names, key="hyb_ch")
        if channel_name:
            X_raw, y, epochs = load_edf(os.path.join(basepath,selected_file), channel_name)
            if X_raw is not None and len(X_raw) > 0:
                subject_id = os.path.basename(selected_file)[:7]
                save_preprocessed_data(subject_id, X_raw, y, epochs, channel_name)
                view_and_save_epochs(X_raw, y, epochs, subject_id, channel_name, key_prefix="HYB")


                if selected_file in storedresults["Hybrid"]:
                    y_pred, y_true = storedresults["Hybrid"][selected_file]
                    st.success("Loaded stored predictions")
                    visualize_predictions(y_true, y_pred, channel_name)
                    plot_per_epoch_hypnogram(np.array(y_true), np.array(y_pred))
                else:
                    input_length = X_raw.shape[1]
                    fe_model = CNNFeatureExtractor(input_length=input_length)
                    fe_model = train_feature_extractor(fe_model, X_raw, epochs=3, batch_size=8, lr=1e-3)
                    fe_model.eval()
                    with torch.no_grad():
                        Xt = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1)
                        flat, _ = fe_model(Xt)
                        cnn_features = flat.numpy()

                    smote = SMOTE(random_state=42)
                    try:
                        X_res, y_res = smote.fit_resample(cnn_features, y)
                    except Exception:
                        X_res, y_res = cnn_features, y

                    rf = RandomForestClassifier().fit(X_res, y_res)
                    y_pred_full = rf.predict(cnn_features)
                    acc = accuracy_score(y, y_pred_full)
                    st.metric("Hybrid Accuracy", f"{acc*100:.2f}%")
                    storedresults["Hybrid"][selected_file] = (y_pred_full, y)
                    visualize_predictions(y, y_pred_full, channel_name)
                    plot_per_epoch_hypnogram(np.array(y), np.array(y_pred_full))
            else:
                st.warning("Could not extract epochs from this file / channel.")
