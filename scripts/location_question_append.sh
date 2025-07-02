# masc_2015
CUDA_LAUNCH_BLOCKING=1 python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=checkpoints/15nt.pt --output_model_path=checkpoints/1.pt --train_info_json=checkpoints/train_info_twitter2015_none_m.json --num_labels=3 --lr=1e-5 --step_size=5 --gamma=0.5 --replay=0
