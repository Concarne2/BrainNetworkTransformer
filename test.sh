#python -m source --multirun datasz=100p model=bnt dataset=ABIDE repeat_time=20 preprocess=mixup

#wait

# python -m source --multirun datasz=100p model=fbnetgen dataset=ABIDE repeat_time=25 preprocess=mixup

# wait

# python -m source --multirun datasz=100p model=brainnetcnn dataset=ABIDE repeat_time=25 preprocess=mixup

# wait

CUDA_VISIBLE_DEVICES=3 python -m source --multirun datasz=100p model=bnt dataset=ABCD repeat_time=4 preprocess=mixup