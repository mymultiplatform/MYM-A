class Config:
    TRAIN_START_DATE = "2018-05-02T08:44:39.292071841Z"
    TRAIN_END_DATE = "2018-05-18T09:27:27.555026009Z"
    TEST_START_DATE = "2018-05-18T09:41:59.385291233Z"
    TEST_END_DATE = "2018-05-18T23:59:43.279929697Z"
    DATA_DIR = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data\test"  # Change to your production data directory
    
    # Model parameters
    SEQUENCE_LENGTH = 60
    PREDICTION_LENGTH = 1440
    BATCH_SIZE = 128
    HIDDEN_SIZE = 256
    NUM_LAYERS = 3
    LEARNING_RATE = 0.001
    EPOCHS = 30  # Increased from 30 to 100
    PATIENCE = 50  # Early stopping patience
    
    # Training parameters
    TRAIN_VAL_SPLIT = 0.8
    NUM_WORKERS = 4
    
    # Other parameters
    RANDOM_SEED = 42
