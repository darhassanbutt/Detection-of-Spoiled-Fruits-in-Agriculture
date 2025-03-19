import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Rescaling, Normalization
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
#help in not showing warning b/c warning make hurdel in output analysis
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Read and show data
data = pd.read_csv('C:\\Users\\butt0\\Downloads\\assignemnt 02ML\\spine.csv')
print('Dimension of given data:', data.shape)
#extract features and split in 80% train and 20% test set
x = data[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11', 'Col12']].values
y = data['Class_label'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=104)
print(x.shape)
# Build the model using the functional API
input_layer = keras.Input(shape=(12,))
#x = Rescaling(scale=1.0 /310)(x)
dense_layer = layers.Dense(150, activation='relu')(input_layer)
#dense_layer = layers.BatchNormalization()(input_layer)
output_layer = layers.Dense(1, activation='sigmoid')(dense_layer)
model = keras.Model(input_layer,output_layer)
# Summarize the model
model.summary()
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train, epochs=30)
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy of traning NN: {accuracy}')
# Predictions
y_pred = (model.predict(x_test) > 0.5).astype("int32")
#print(y_pred)
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)
# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy of Testing NN: {acc}')
linear = LogisticRegression()
linear.fit(x_train,y_train)
accuracy_train = accuracy_score(y_train, linear.predict(x_train))
accuracy_test = accuracy_score(y_test,linear.predict(x_test))
print("Training Accuracy of built in logisticregression:", accuracy_train)
print("Test Accuracy of built in logisticregression:", accuracy_test)
mlp = MLPClassifier(hidden_layer_sizes=(1))
mlp.fit(x_train, y_train.ravel())
print("built ,NN Accuracy ", accuracy_score(y_test,mlp.predict(x_test)))


