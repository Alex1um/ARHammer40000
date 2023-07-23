VIDEO_CAPTURE_DEVICE = 2
# VIDEO_CAPTURE_DEVICE = "http://localhost:8081/"
ROBOT_ADDRESS = "10.0.0.155"
# ROBOT_ADDRESS = "10.0.0.144"
RESOLUTION = (640, 480)

MS = 115  # moving speed
TS = 130  # turning speed
SEA = 30  # small enough angle, when stop aiming
BEA = 45  # big enough angle, when start reaiming


GRID_WIDTH = 6
GRID_HEIGHT = 6
INITIAL_ID = 8

# Training parameters
N_TRAINING_EPISODES = 30000  # TOTAL TRAINING EPISODES
LEARNING_RATE = 0.7          # LEARNING RATE

# Evaluation parameters
N_EVAL_EPISODES = 100        # TOTAL NUMBER OF TEST EPISODES

# Environment parameters
ENV_ID = "FrozenLake-v1"     # Name of the environment
MAX_STEPS = 99               # MAX STEPS PER EPISODE
GAMMA = 0.95                 # DISCOUNTING RATE
EVAL_SEED = []               # THE EVALUATION SEED OF THE ENVIRONMENT

# Exploration parameters
MAX_EPSILON = 1.0             # EXPLORATION PROBABILITY AT START
MIN_EPSILON = 0.05            # MINIMUM EXPLORATION PROBABILITY
DECAY_RATE = 0.0005            # EXPONENTIAL DECAY RATE FOR EXPLORATION PROB
NEXT_GRID_POINT = None
