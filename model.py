import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load the sequence dataset
data = np.load('flood_prediction_sequences_5.npz')
X, y = data['X'], data['y']

# Reshape and scale the data
n_features = X.shape[2]
scaler = MinMaxScaler()
X_reshaped = X.reshape(-1, n_features)
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(X.shape)

# Split the data into training and test sets
split = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# Save the trained model and scaler
model.save('flood_model_5.keras')
with open('scaler_5.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Test the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model accuracy: {accuracy}")
