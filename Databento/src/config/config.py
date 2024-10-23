# config/config.py
from datetime import datetime, timezone, timedelta

class Config:
    # Initialize date attributes
    TRAIN_START_DATE = None
    TRAIN_END_DATE = None
    TEST_START_DATE = None
    TEST_END_DATE = None
    
    # Model parameters
    SEQUENCE_LENGTH = 60
    PREDICTION_LENGTH = 1440
    BATCH_SIZE = 128
    HIDDEN_SIZE = 256
    NUM_LAYERS = 3
    LEARNING_RATE = 0.001
    EPOCHS = 30
    PATIENCE = 50
    
    # Training parameters
    TRAIN_VAL_SPLIT = 0.8
    NUM_WORKERS = 4
    
    # Other parameters
    RANDOM_SEED = 42
    DATA_DIR = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data\test"

    @staticmethod
    def parse_date(date_str):
        """Parse date string to datetime object."""
        try:
            # First try parsing with 9 decimal places
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                # If that fails, try truncating to 6 decimal places
                truncated_date = date_str[:26] + 'Z'  # Keep only up to 6 decimal places
                return datetime.strptime(truncated_date, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
            except ValueError as e:
                print(f"Error parsing date: {date_str}")
                raise e

    @classmethod
    def set_date_range(cls, start_date_str, end_date_str):
        """Set the date ranges for training and testing periods."""
        # Parse dates
        start_date = cls.parse_date(start_date_str)
        end_date = cls.parse_date(end_date_str)
        
        # Calculate periods
        total_period = end_date - start_date
        training_days = int(total_period.days * 0.8)  # 80% for training
        
        # Calculate boundaries
        train_end = start_date + timedelta(days=training_days)
        test_start = train_end + timedelta(minutes=15)  # 15-minute gap between train and test
        
        # Format dates with proper precision
        cls.TRAIN_START_DATE = start_date_str  # Keep original string format
        cls.TRAIN_END_DATE = train_end.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"  # Match precision
        cls.TEST_START_DATE = test_start.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"  # Match precision
        cls.TEST_END_DATE = end_date_str  # Keep original string format
        
        # Print information
        print("\nData Periods:")
        print(f"Total Period: {start_date} to {end_date}")
        print(f"Training Period: {start_date} to {train_end}")
        print(f"Testing Period: {test_start} to {end_date}")
        
        return cls.TRAIN_START_DATE, cls.TRAIN_END_DATE, cls.TEST_START_DATE, cls.TEST_END_DATE
# Usage in main.py (add at the beginning of main())
def initialize_config():
    start_date = "2018-05-02T08:44:39.292071841Z"
    end_date = "2018-05-18T09:27:27.555026009Z"
    Config.set_date_range(start_date, end_date)