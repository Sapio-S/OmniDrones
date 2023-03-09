import wandb
run = wandb.init()
artifact = run.use_artifact('linster/gpu-onpolicy/source-gpu-onpolicy-examples_demo_prey.py:v54', type='code')
artifact_dir = artifact.download()