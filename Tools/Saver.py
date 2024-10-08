import os
import joblib  # or use pickle

# Create a directory named 'Savers' on the Desktop
savers_dir = "C:/Users/ds1020254/Desktop/Savers"
os.makedirs(savers_dir, exist_ok=True)

# Save the LSTM model
lstm_model_file = os.path.join(savers_dir, "lstm_model.pkl")
joblib.dump(model, lstm_model_file)  # Use `pickle.dump(model, open(lstm_model_file, 'wb'))` if using pickle

# Save the RL training data
rl_training_data_file = os.path.join(savers_dir, "rl_training_data.pkl")
joblib.dump(trade_log, rl_training_data_file)  # Use `pickle.dump(trade_log, open(rl_training_data_file, 'wb'))` if using pickle

print("Files saved successfully!")
