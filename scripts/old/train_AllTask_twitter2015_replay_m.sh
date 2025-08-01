source /d/ProgramData/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --num_labels=3 --mode=text_only --epoch=5 --lr=1e-5 --step_size=2 --gamma=0.1 --replay=1
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --replay=1 --epoch=5
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --replay=1 --epoch=5
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --replay=1 --epoch=5 --lr=5e-5 --step_size=10 --gamma=0.5
# mnre_2015
python -m scripts.train --task_name=mnre --session_name=mnre_1 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MNRE/twitter2015/train.txt --test_text_file=data/MNRE/twitter2015/test.txt --dev_text_file=data/MNRE/twitter2015/dev.txt --num_labels=23 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --replay=1 --epoch=5



# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/2.pt --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --num_labels=3 --epoch=5 --lr=1e-5 --step_size=2 --gamma=0.1 --replay=1
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=checkpoints/2.pt --output_model_path=checkpoints/2.pt --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --replay=1 --epoch=5
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=checkpoints/2.pt --output_model_path=checkpoints/2.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --replay=1 --epoch=5
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=checkpoints/2.pt --output_model_path=checkpoints/2.pt --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --replay=1 --epoch=5 --lr=5e-5 --step_size=10 --gamma=0.5
# mnre_2015
python -m scripts.train --task_name=mnre --session_name=mnre_2 --pretrained_model_path=checkpoints/2.pt --output_model_path=checkpoints/2.pt --train_text_file=data/MNRE/twitter2015/train.txt --test_text_file=data/MNRE/twitter2015/test.txt --dev_text_file=data/MNRE/twitter2015/dev.txt --num_labels=23 --train_info_json=checkpoints/train_info_twitter2015_replay_m.json --replay=1 --epoch=5
