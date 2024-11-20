import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import tensorflow as tf
from model import irisPrediction
from model import splitting

keras = tf.keras
models = tf.keras.models
layers = tf.keras.layers


#FIXING THE UI
st.markdown(
    """
    <style>
    .block-container {
	 		width: 100%;
			height: 100%;
			padding-top: 10px;
			padding-bottom: 10px;
			padding-left: 10px;
			padding-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title('Predict Your Iris!')
st.subheader('adit is botanis, bantu adit prediksi tipe iris apakah yang ia punya 🤔')


# Loading Data
data = pd.read_csv('clean_dataset.csv')
data.drop(columns=['Unnamed: 0'], inplace=True)


# train test split
X_train, y_train, X_test, y_test = splitting(data, 0.65, 'species')


# intializing the K, later let's make it customized!
k = 11


# make instance
iris_prediction = irisPrediction(X_train, y_train, X_test, y_test, k)


# making y_pred
y_pred = iris_prediction.predict(X_train, y_train, X_test, k)


#input fields for features
sepalLength = st.number_input('Sepal Length (CM): ', min_value=0.0, step=0.1)
sepalWidth = st.number_input('Sepal Width (CM): ', min_value=0.0, step=0.1)
petalLength = st.number_input('Petal Length (CM): ', min_value=0.0, step=0.1)
petalWidth = st.number_input('Petal Width (CM): ', min_value=0.0, step=0.1)


#Predicting by clicking the 'predict button'
if st.button('predict'):
	#input data in the correct format which is using numpy array
	input_data = np.array([sepalLength, sepalWidth, petalLength, petalWidth])

	#predicting to the the species of the specified data size
	st.write({iris_prediction.accuracy(y_test, y_pred)})
	st.write(f'the species is {iris_prediction.predict_single(X_train=X_train, y_train=y_train, new_instance=input_data, k=k)}')


st.write(f'The data shape: {data.shape}')
st.table(data)












# #MAKING SELECT BOX OF PLOTS
# choose_plot = st.selectbox(
# 	'Choose plot',
# 	('Pairplot','Barplot','Boxplot')
# )

# if choose_plot == 'Pairplot': 
# 	st.write('Pairplot')
# 	fig = sns.pairplot(data, hue='species')
# 	st.pyplot(fig)

# elif choose_plot == 'Barplot':
# 	st.write('Barplot')
# 	fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(10,50))
# 	sns.barplot(x='species', y='sepal_length', hue='species', data=data, ax=ax1)
# 	plt.title('sepal_length')
# 	sns.barplot(x='species', y='sepal_width', hue='species', data=data, ax=ax2)
# 	plt.title('sepal_width')
# 	sns.barplot(x='species', y='petal_length', hue='species', data=data, ax=ax3)
# 	plt.title('petal_length')
# 	sns.barplot(x='species', y='petal_width', hue='species', data=data, ax=ax4)
# 	plt.title('petal_width')
# 	st.pyplot(fig)

# elif choose_plot == 'Boxplot':
# 	st.write('Boxplot')
# 	fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(10,50))
# 	sns.boxplot(x='species', y='sepal_length', hue='species', data=data, ax=ax1)
# 	plt.title('sepal_length')
# 	sns.boxplot(x='species', y='sepal_width', hue='species', data=data, ax=ax2)
# 	plt.title('sepal_width')
# 	sns.boxplot(x='species', y='petal_length', hue='species', data=data, ax=ax3)
# 	plt.title('petal_length')
# 	sns.boxplot(x='species', y='sepal_width', hue='species', data=data, ax=ax4)
# 	plt.title('petal_width')
# 	st.pyplot(fig)

