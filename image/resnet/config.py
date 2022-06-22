from colossalai.amp import AMP_TYPE

BATCH_SIZE = 16384
NUM_EPOCHS = 200

CONFIG = dict(fp16=dict(mode=AMP_TYPE.TORCH))
