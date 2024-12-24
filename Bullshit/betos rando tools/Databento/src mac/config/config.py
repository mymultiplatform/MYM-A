class Config:
    TRAIN_START_DATE = "2018-05-02T08:44:39.292059872Z"
    TRAIN_END_DATE = "2018-05-11T08:00:02.198789657Z"
    TEST_START_DATE = "2018-05-11T08:00:02.198789657Z"
    TEST_END_DATE = "2018-05-11T23:59:57.298138119Z"
    DATA_DIR = "/Users/jazzhashzzz/Desktop/data for scripts/data bento data/test"  # Change to your production data directory
    
    # System Resource Configuration
    CPU_CORES = 10  # Your MacBook's core count
    MEMORY_GB = 16  # Your total RAM in GB
    
    # Optimized Model Parameters
    SEQUENCE_LENGTH = 720     # Reduced from 1440 to save memory
    PREDICTION_LENGTH = 20    # Keep as is
    BATCH_SIZE = 256         # Increased from 128 for better CPU utilization
    HIDDEN_SIZE = 512        # Increased for more model capacity
    NUM_LAYERS = 4           # Keep complex architecture since we have resources
    
    # Training Parameters
    LEARNING_RATE = 0.001
    EPOCHS = 1              # Increased from 1 for proper training
    PATIENCE = 20           # Reduced from 50 for faster training cycles
    
    # Resource Optimization Parameters
    NUM_WORKERS = 8         # 80% of CPU cores (10 cores * 0.8)
    TRAIN_VAL_SPLIT = 0.8
    PIN_MEMORY = True       # Enable memory pinning
    PERSISTENT_WORKERS = True
    
    # Performance Optimization Flags
    CUDNN_BENCHMARK = True
    USE_FLOAT32 = True      # Use float32 instead of float64
    
    # Memory Management
    PREFETCH_FACTOR = 2     # Control memory usage for data loading
    MAX_MEMORY_ALLOCATED = int(0.7 * MEMORY_GB * 1024 * 1024 * 1024)  # 70% of total RAM
    
    # Data Processing
    PARALLEL_PROCESSING = True
    PROCESS_CHUNK_SIZE = 10000  # Number of rows to process at once
    
    # Random Seed
    RANDOM_SEED = 42

    @classmethod
    def print_resource_usage(cls):
        """Print current system resource usage"""
        import psutil
        import torch
        
        print("\nSystem Resource Usage:")
        print(f"CPU Cores Available: {cls.CPU_CORES}")
        print(f"CPU Usage: {psutil.cpu_percent()}%")
        print(f"Total Memory: {cls.MEMORY_GB} GB")
        print(f"Memory Usage: {psutil.virtual_memory().percent}%")
        if torch.cuda.is_available():
            print(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    @classmethod
    def optimize_for_system(cls):
        """Automatically adjust parameters based on system load"""
        import psutil
        
        # Adjust workers based on current CPU usage
        current_cpu_usage = psutil.cpu_percent()
        if current_cpu_usage > 80:
            cls.NUM_WORKERS = max(4, cls.CPU_CORES // 2)
        
        # Adjust batch size based on available memory
        available_memory = psutil.virtual_memory().available
        if available_memory < (4 * 1024 * 1024 * 1024):  # Less than 4GB available
            cls.BATCH_SIZE = 128
            cls.PREFETCH_FACTOR = 1
        
        # Print optimized settings
        print("\nOptimized Settings:")
        print(f"Number of Workers: {cls.NUM_WORKERS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Prefetch Factor: {cls.PREFETCH_FACTOR}")