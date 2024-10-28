class Config:
    TRAIN_START_DATE = "2018-05-02T08:44:39.292059872Z"
    TRAIN_END_DATE = "2018-05-18T09:27:27.555011410Z"
    TEST_START_DATE = "2018-05-18T09:27:27.555011410Z"
    TEST_END_DATE = "2018-05-18T23:59:43.279917301Z"
    DATA_DIR = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data\test"  # Change to your production data directory
    
    # Model parameters
    SEQUENCE_LENGTH = 1440
    PREDICTION_LENGTH = 60
    BATCH_SIZE = 128
    HIDDEN_SIZE = 256
    NUM_LAYERS = 4
    LEARNING_RATE = 0.001
<<<<<<< HEAD
    EPOCHS = 50 # Increased from 30 to 100
=======
    EPOCHS = 1# Increased from 30 to 100
>>>>>>> 7635e0eb9f8de1894af1b357848f0adcfc8d1e7b
    PATIENCE = 50  # Early stopping patience
    
    # Training parameters
    TRAIN_VAL_SPLIT = 0.8
    NUM_WORKERS = 4
    
    # Other parameters
    RANDOM_SEED = 42
