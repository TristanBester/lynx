export WANDB_PROJECT=sweep-testing
export SWEEP_ID=96e66xia
export ENTITY=tristanbester1

echo $ENTITY/$WANDB_PROJECT/$SWEEP_ID

wandb agent $ENTITY/$WANDB_PROJECT/$SWEEP_ID
