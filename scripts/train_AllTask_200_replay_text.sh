source /d/ProgramData/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=checkpoints/model_mner.pt --output_model_path=checkpoints/model_mabsa.pt --train_text_file=data/MASC/twitter2015/train__.txt --test_text_file=data/MASC/twitter2015/test__.txt --dev_text_file=data/MASC/twitter2015/dev__.txt --image_dir=data/MASC/twitter2015/images --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_replay_text.json --replay=1  --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5
