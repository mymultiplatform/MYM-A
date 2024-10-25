class Config:
    TRAIN_START_DATE = "2018-05-02T08:44:39.292071841Z"
    TRAIN_END_DATE = "2018-05-10T23:59:56.173292547Z"
    TEST_START_DATE = "2018-05-10T23:59:56.173292547Z"
    TEST_END_DATE = "2018-05-11T23:59:57.298151284Z"
    DATA_DIR = "/Users/jazzhashzzz/Desktop/data for scripts/data bento data/test"  # Change to your production data directory
    
    # Model parameters
    SEQUENCE_LENGTH = 1440
    PREDICTION_LENGTH = 60
    BATCH_SIZE = 64
    HIDDEN_SIZE = 512
    NUM_LAYERS = 4
    LEARNING_RATE = 0.005
    EPOCHS = 1  # Increased from 30 to 100
    PATIENCE = 200  # Early stopping patience
    
    # Training parameters
    TRAIN_VAL_SPLIT = 0.8
    NUM_WORKERS = 4
    
    # Other parameters
    RANDOM_SEED = 1000
