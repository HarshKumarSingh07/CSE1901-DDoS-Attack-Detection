import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@st.cache(persist=True)
def load_data():
    data = pd.read_csv('udp_attack.csv')
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data


@st.cache(persist=True)
def split(df):
    service_values = np.unique(df.loc[:,"service"])
    print(service_values)
    mid = (len(service_values)+1)/2
    print(mid)
    for i in range(len(service_values)):
        df = df.replace(service_values[i], (i-mid)/10)
    y = df.type
    x = df.drop(columns=['type'])
    x = x.loc[:,["dst_bytes","service","src_bytes","dst_host_srv_count","count"]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test


def plot_metrics(metrics_list, model, x_test, y_test, class_names):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()