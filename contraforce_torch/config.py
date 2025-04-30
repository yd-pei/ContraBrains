# Hyperparameters and training settings
LEARNING_RATE      = 2.5e-4
GAMMA              = 0.99
TAU                = 0.95
CLIP_PARAM         = 0.2

UPDATE_INTERVAL    = 128
BATCH_SIZE         = 32
PPO_EPOCHS         = 4

SCORE_SCALE        = 1.0
ENTROPY_BONUS      = 0.02
PIXEL_SCALE        = 0.05
JUMP_PENALTY       = 0.01

DEFAULT_REPEAT     = 4
JUMP_REPEAT        = 12

STACK_SIZE         = 4
RENDER             = True
MAX_TIMESTEPS      = 1_000_000
