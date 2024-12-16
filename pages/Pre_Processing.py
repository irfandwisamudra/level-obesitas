import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from component.nav import navbar
from component.bootstrap import bootstrap
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

# ------------------navbar-----------------------
navbar()

def deskPre() :
    bootstrap()
    desk_pre = """
    <p>
        Preprocessing data adalah proses untuk mengubah data mentah menjadi lebih teratur agar ketika dilakukan teknik data mining tingkat akurasinya lebih tinggi.
    </p>
    <p>
        Data cleaning atau pembersihan data terutama dilakukan sebagai bagian dari data preprocessing untuk membersihkan data dengan mengisi nilai yang hilang, menghaluskan data yang noise, menyelesaikan data yang tidak konsisten, dan menghapus outlier.
    </p>
    """
    st.markdown(desk_pre,unsafe_allow_html=True)


def pre():
    bootstrap()
    # ------------------FULL DATA-----------------------
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

    # Descriptive Statistics
    st.subheader('Descriptive Statistics')
    st.write(df_main.describe())

    # Check for missing values
    st.subheader('Missing Values')
    st.write(df_main.isnull().sum())

    # Checking for outliers using boxplots
    st.subheader('Checking Outliers')
    continuous_columns = [col for col in df.columns if len(df[col].unique())>10]
    for col in continuous_columns:
        st.write(f"Boxplot for {col}")
        plt.figure()
        sns.boxplot(df_main[col])
        plt.title(f'Distribution for {col}')
        st.pyplot(plt)

    # Handle outliers by removing the maximum values for weight and height
    st.subheader('Outlier Treatment')
    df_main.drop(df_main[df_main['Weight'] == df_main['Weight'].max()].index, inplace=True)
    df_main.drop(df_main.nlargest(2, 'Height').index, inplace=True)

    # Recheck outliers
    for col in continuous_columns:
        st.write(f"Boxplot for {col} after outlier treatment")
        plt.figure()
        sns.boxplot(df_main[col])
        plt.title(f'Distribution for {col}')
        st.pyplot(plt)

    # Prepare the feature and target variables
    features = X_df.columns  # Assuming X_df columns are the feature names
    target = y_df.columns    # Assuming y_df columns are the target variable

    df_prep = df_main[features]  # Features for training
    df_prep.to_csv('obesity-levels_prep.csv', index=False)

    st.subheader('Data Preparation')
    st.write(df_prep)

    # Apply MinMaxScaler to the features
    scaler = MinMaxScaler()
    x_prep = scaler.fit_transform(df_prep)

    # Save the scaler for later use
    joblib.dump(scaler, 'scaler.pkl')
    st.write("Scaler telah disimpan sebagai 'scaler.pkl'.")

    st.subheader('Data after MinMax Scaling')
    st.write(x_prep)


# ------------------PreProcessing Data-----------------------
st.markdown(
    '''
    <h1 align="center">PRE PROCESSING DATA</h1>
    '''
    , unsafe_allow_html=True
)
deskPre()

st.subheader('Preprocessing')
st.code("""
    # Load and preprocess the dataset
    df_main = df.copy()

    # Perform label encoding on categorical columns
    for col in categorical_columns:
        le = LabelEncoder()
        df_main[col] = le.fit_transform(df_main[col])

    # Descriptive Statistics
    df_main.describe()

    # Check missing values
    df_main.isnull().sum()

    # Handle outliers by removing max values
    df_main.drop(df_main[df_main['Weight'] == df_main['Weight'].max()].index, inplace=True)
    df_main.drop(df_main.nlargest(2, 'Height').index, inplace=True)

    # Apply MinMaxScaler to features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_main[features])

    # Save scaler for future use
    joblib.dump(scaler, 'scaler.pkl')
    """)

pre()
