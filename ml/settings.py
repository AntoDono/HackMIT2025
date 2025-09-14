# =============================================================================
# TRAINING PARAMETERS - Configure these at the top
# =============================================================================
CHUNK_SIZE = 30                    # Size of each data chunk for training
EPOCHS = 100                        # Number of training epochs
BATCH_SIZE = 64                    # Training batch size
VALIDATION_SPLIT = 0.2             # Fraction of data for validation
LEARNING_RATE = 0.001              # Learning rate for optimizer
DROPOUT_RATE = 0.05                 # Dropout rate for regularization
HIDDEN_UNITS = [512, 1024, 2048, 1024, 512]       # Hidden layer sizes
TEST_SIZE = 0.2                    # Fraction of data for testing
RANDOM_STATE = 42                  # Random seed for reproducibility

# Data paths
DATA_DIR = '../backend/saved-audio'
MODEL_SAVE_PATH = '../models/brainwave_model.h5'
ENCODER_SAVE_PATH = '../models/label_encoder.joblib'

# Brainwave feature columns
FEATURE_COLUMNS = [
    'attention', 'meditation', 'delta', 'theta', 
    'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta'
]

# Hardcoded emotion labels
HARDCODED_LABELS = ['focused', 'nervous', 'relax', 'stress', 'surprise']
