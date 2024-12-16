import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

# ------------------navbar-----------------------
from component.nav import navbar
navbar()

def build_lstm_model(x_train, y_train, x_test, y_test):
    # Reshape data for LSTM (3D: samples, timesteps, features)
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    # LSTM model creation
    model = Sequential([
        LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stopping], verbose=1)

    # Prediction
    y_pred = model.predict(x_test)
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_test_classes = tf.argmax(y_test, axis=1)

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    st.write("Confusion Matrix for LSTM:")
    st.write(cm)

    st.write("Classification Report for LSTM:")
    st.write(classification_report(y_test_classes, y_pred_classes))

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    st.write(f"LSTM Accuracy: {accuracy * 100:.2f}%")

    # Accuracy plot
    fig, ax = plt.subplots()  # Create figure and axes
    ax.plot(history.history['accuracy'], label='Training Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.legend()
    ax.set_title('Training and Validation Accuracy')
    st.pyplot(fig)  # Pass the figure to st.pyplot()

    fig, ax = plt.subplots()  # Create another figure for loss
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.legend()
    ax.set_title('Training and Validation Loss')
    st.pyplot(fig)  # Pass the figure to st.pyplot()

    # Save the model
    model.save('lstm_model.keras')

    return model

def random_forest_model(x_train, y_train, x_test, y_test):
    # Train a Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)

    # Predictions and evaluation
    y_pred_rf = rf_model.predict(x_test)
    
    # Convert y_test and y_pred_rf from one-hot encoding (if they are one-hot encoded) to class indices
    if len(y_test.shape) > 1:
        y_test_classes = np.argmax(y_test, axis=1)  # Convert one-hot to class labels
    else:
        y_test_classes = y_test  # If it's already an integer label, no conversion needed
    
    if len(y_pred_rf.shape) > 1:
        y_pred_classes = np.argmax(y_pred_rf, axis=1)  # Convert one-hot to class labels
    else:
        y_pred_classes = y_pred_rf  # If it's already an integer label, no conversion needed

    # Confusion Matrix and Classification Report
    cm_rf = confusion_matrix(y_test_classes, y_pred_classes)
    st.write("Confusion Matrix for Random Forest:")
    st.write(cm_rf)

    st.write("Classification Report for Random Forest:")
    st.write(classification_report(y_test_classes, y_pred_classes))

    accuracy_rf = accuracy_score(y_test_classes, y_pred_classes)
    st.write(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

    # Save model
    joblib.dump(rf_model, 'rf_model.joblib')

def knn_model(x_train, y_train, x_test, y_test):
    # Train a KNN model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(x_train, y_train)

    # Predictions and evaluation
    y_pred_knn = knn_model.predict(x_test)
    
    # Convert y_test and y_pred_knn from one-hot encoding (if they are one-hot encoded) to class indices
    if len(y_test.shape) > 1:
        y_test_classes = np.argmax(y_test, axis=1)  # Convert one-hot to class labels
    else:
        y_test_classes = y_test  # If it's already an integer label, no conversion needed
    
    if len(y_pred_knn.shape) > 1:
        y_pred_classes = np.argmax(y_pred_knn, axis=1)  # Convert one-hot to class labels
    else:
        y_pred_classes = y_pred_knn  # If it's already an integer label, no conversion needed

    # Confusion Matrix and Classification Report
    cm_knn = confusion_matrix(y_test_classes, y_pred_classes)
    st.write("Confusion Matrix for KNN:")
    st.write(cm_knn)

    st.write("Classification Report for KNN:")
    st.write(classification_report(y_test_classes, y_pred_classes))

    accuracy_knn = accuracy_score(y_test_classes, y_pred_classes)
    st.write(f"KNN Accuracy: {accuracy_knn * 100:.2f}%")

    # Save model
    joblib.dump(knn_model, 'knn_model.joblib')

# ------------------Main Streamlit App----------------------

def model():
    # ------------------Modelling-----------------------
    st.markdown('<h1 align="center">Modelling</h1>', unsafe_allow_html=True)

    # Load the obesity levels dataset
    estimation_of_obesity_levels = fetch_ucirepo(id=544)
    data = estimation_of_obesity_levels.data
    X = data.features
    y = data.targets

    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=estimation_of_obesity_levels.features)
    y_df = pd.DataFrame(y, columns=estimation_of_obesity_levels.targets)

    # Gabungkan fitur dan target
    df = pd.concat([X_df, y_df], axis=1)

    # Perform label encoding for categorical columns (if any)
    df_main = df.copy()
    categorical_columns = [col for col in df.columns if df[col].dtype == 'O']

    for col in categorical_columns:
        le = LabelEncoder()
        df_main[col] = le.fit_transform(df_main[col])
    
    # Handle outliers by removing the maximum values for weight and height
    df_main.drop(df_main[df_main['Weight'] == df_main['Weight'].max()].index, inplace=True)
    df_main.drop(df_main.nlargest(2, 'Height').index, inplace=True)

    # Prepare the feature and target variables
    features = X_df.columns  # Assuming X_df columns are the feature names
    target = y_df.columns    # Assuming y_df columns are the target variable

    df_prep = df_main[features]  # Features for training

    # Apply MinMaxScaler to the features
    scaler = MinMaxScaler()
    x = scaler.fit_transform(df_prep)
    y = df_main[target]

    # Train test split of data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Convert y_train and y_test to categorical if necessary
    if len(y_train.shape) == 1 or y_train.shape[1] == 1:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    # LSTM Model
    st.subheader("LSTM Model")
    build_lstm_model(x_train, y_train, x_test, y_test)

    # Random Forest Model
    st.subheader("Random Forest Model")
    random_forest_model(x_train, y_train, x_test, y_test)

    # KNN Model
    st.subheader("KNN Model")
    knn_model(x_train, y_train, x_test, y_test)

# Run the model function
model()
