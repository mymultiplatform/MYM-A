class Config:
    TRAIN_START_DATE = "2018-05-02T08:44:39.292071841Z"
    TRAIN_END_DATE = "2024-10-21T08:00:00.143539165Z"
    TEST_START_DATE = "2024-10-21T08:00:00.143539165Z"
    TEST_END_DATE = "2024-10-21T23:59:51.581344604Z"
    DATA_DIR = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data\NVDA"  # Change to your production data directory
    
    # Model parameters
    SEQUENCE_LENGTH = 1440
    PREDICTION_LENGTH = 60
    BATCH_SIZE = 128
    HIDDEN_SIZE = 256
    NUM_LAYERS = 3
    LEARNING_RATE = 0.001
    EPOCHS = 1# Increased from 30 to 100
    PATIENCE = 50  # Early stopping patience
    
    # Training parameters
    TRAIN_VAL_SPLIT = 0.8
    NUM_WORKERS = 4
    
    # Other parameters
    RANDOM_SEED = 42
