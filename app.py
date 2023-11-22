import streamlit as st
import numpy as np
import utils
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost
import pickle
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score,roc_curve, confusion_matrix,plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, recall_score



st.set_option('deprecation.showPyplotGlobalUse', False)



def main():
    st.title("DDoS Attack Detection (CSE1901 J component)")
    
    st.sidebar.title("Prof: Dr. Suresh. A sir")
    st.markdown("Is your incoming attack is normal or dangerous?")
    st.markdown("Harsh Kumar Singh (20BCI0290)")
    st.markdown("Ankit Mahajan (20BCE2381)")
    st.markdown("Tathagat Gaur (20BCE2246)")
    st.sidebar.markdown("Is your incoming attack is normal or dangerous?")



    
    df = pd.read_csv("./dataset.csv",index_col=0)
    def check_service(service):
        serv = []
        for i in service:
            if i=="private":
                ser = 0.0
                serv.append(ser)
            elif i=="domain_u":
                ser = -0.3
                serv.append(ser)
            elif i=="other":
                ser = -0.1
                serv.append(ser)
            elif i=="ntp_u":
                ser = -0.2
                serv.append(ser)
            else:
                ser = 0.1
                serv.append(ser)

        return serv
        
    df["service"] = check_service(df["service"])
    features = ["service","src_bytes","dst_bytes","wrong_fragment","count","num_compromised","srv_count","dst_host_srv_count","dst_host_diff_srv_rate"]
    target = "result"
    x = df.loc[:,features]
    y = df.loc[:,target]
    classes = np.unique(y)

    for i in range(len(classes)):
        if i == 1:
            df = df.replace(classes[i], 0)
        else:
            df = df.replace(classes[i], 1)

    #turning the service attribute to categorical values
    df=df.replace("eco_i",-0.1)
    df=df.replace("ecr_i",0.0)
    df=df.replace("tim_i",0.1)
    df=df.replace("urp_i",0.2)
    y = df.loc[:,target]
    x = x.loc[:,["dst_bytes","service","src_bytes","dst_host_srv_count","count"]]
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3)

    
    class_names = ["normal", "attack"]

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression",
                                                     "SGD","Tree-based Stacked Generalization"))


    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)


    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='Lr')
        max_iter = st.sidebar.slider("Maximum no. of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)


    if classifier == 'Tree-based Stacked Generalization':
        st.sidebar.subheader("Model Hyperparameters are default")
        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Tree-based Stacked Generalization Results")
            level0 = list()
            level0.append(('rf_clf', RandomForestClassifier()))
            level0.append(('ext_clf', ExtraTreesClassifier()))
            level0.append(('xgb_clf', xgboost.XGBClassifier()))
            level0.append(('lgb_clf', LGBMClassifier()))
            level0.append(('ada_clf', AdaBoostClassifier()))
            level1 = CatBoostClassifier()
            model = StackingClassifier(estimators=level0, final_estimator=level1, cv=2)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)

    if classifier == 'SGD':
        st.sidebar.subheader("Model Hyperparameters are default")
        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))
        

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("SGD Results")
            model = RandomForestClassifier()
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)


    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("SDN Dataset (Classification)")
        st.write(df)

    title = st.text_input('Give your feedback' )
if __name__ == '__main__':
    main()