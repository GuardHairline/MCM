source /d/ProgramData/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train__.txt --test_text_file=data/MASC/twitter2015/test__.txt --dev_text_file=data/MASC/twitter2015/dev__.txt --train_info_json=checkpoints/train_info_twitter2015-200_replay_multi.json --num_labels=3 --fusion_strategy=concat --epoch=5 --lr=1e-5 --step_size=2 --gamma=0.1 --replay=1
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train__.txt --test_text_file=data/MASC/twitter2015/test__.txt --dev_text_file=data/MASC/twitter2015/dev__.txt --num_labels=3 --fusion_strategy=concat --train_info_json=checkpoints/train_info_twitter2015-200_replay_multi.json --replay=1
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train__.txt --test_text_file=data/MNER/twitter2015/test__.txt --dev_text_file=data/MNER/twitter2015/dev__.txt --num_labels=9 --fusion_strategy=concat --train_info_json=checkpoints/train_info_twitter2015-200_replay_multi.json --replay=1
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train__.txt --test_text_file=data/MASC/twitter2015/test__.txt --dev_text_file=data/MASC/twitter2015/dev__.txt --num_labels=7 --fusion_strategy=concat --train_info_json=checkpoints/train_info_twitter2015-200_replay_multi.json --replay=1 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5
# mnre_2015
python -m scripts.train --task_name=mnre --session_name=mnre_1 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/15ret.pt --train_text_file=data/MNRE/twitter2015/train__.txt --test_text_file=data/MNRE/twitter2015/test__.txt --dev_text_file=data/MNRE/twitter2015/dev__.txt --num_labels=23 --fusion_strategy=concat --train_info_json=checkpoints/train_info_twitter2015-200_replay_multi.json --replay=1
