# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=checkpoints/15nt.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_info_json=checkpoints/train_info_twitter2015_none_m.json --num_labels=3 --lr=1e-5 --step_size=5 --gamma=0.5 --replay=0
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5
# mnre_2015
python -m scripts.train --task_name=mnre --session_name=mnre_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=checkpoints/15nm.pt --train_text_file=data/MNRE/twitter2015/train.txt --test_text_file=data/MNRE/twitter2015/test.txt --dev_text_file=data/MNRE/twitter2015/dev.txt --num_labels=23 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0

# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=checkpoints/15rt.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --num_labels=3 --epoch=5 --lr=1e-5 --step_size=2 --gamma=0.1 --replay=1
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --replay=1
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --replay=1
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --replay=1 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5
# mnre_2015
python -m scripts.train --task_name=mnre --session_name=mnre_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=checkpoints/15rm.pt --train_text_file=data/MNRE/twitter2015/train.txt --test_text_file=data/MNRE/twitter2015/test.txt --dev_text_file=data/MNRE/twitter2015/dev.txt --num_labels=23 --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --replay=1

# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=checkpoints/15ret.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_info_json=checkpoints/train_info_twitter2015_replayEwc_m.json --num_labels=3 --epoch=5 --lr=1e-5 --step_size=2 --gamma=0.1 --replay=1 --ewc=1 --ewc_dir=/root/autodl-tmp/ewc_params
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_replayEwc_m.json --replay=1 --ewc=1 --ewc_dir=/root/autodl-tmp/ewc_params
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015_replayEwc_m.json --replay=1 --ewc=1 --ewc_dir=/root/autodl-tmp/ewc_params
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015_replayEwc_m.json --replay=1 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5 --ewc=1 --ewc_dir=/root/autodl-tmp/ewc_params
# mnre_2015
python -m scripts.train --task_name=mnre --session_name=mnre_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=checkpoints/15rem.pt --train_text_file=data/MNRE/twitter2015/train.txt --test_text_file=data/MNRE/twitter2015/test.txt --dev_text_file=data/MNRE/twitter2015/dev.txt --num_labels=23 --train_info_json=checkpoints/train_info_twitter2015_replayEwc_m.json --replay=1 --ewc=1 --ewc_dir=/root/autodl-tmp/ewc_params
