##### 17+15


## replay
#
## masc_2017
#python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_replay.json --lr=1e-5 --step_size=5 --gamma=0.5 --replay=1
## mate_2017
#python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_replay.json --replay=1
## mner_2017
#python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --num_labels=9 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_replay.json --replay=1
## mabsa_2017
#python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_replay.json --replay=1
#
## masc_2017
#python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_and_2017_replay.json --lr=1e-5 --step_size=5 --gamma=0.5 --replay=1
## mate_2017
#python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_and_2017_replay.json --replay=1
## mner_2017
#python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015_and_2017_replay.json --replay=1
# mabsa_2017
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1517replay.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015_and_2017_replay.json --replay=1


# none

# masc_2017
python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_none.json --lr=1e-5 --step_size=5 --gamma=0.5 --replay=0
# mate_2017
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_none.json --replay=0
# mner_2017
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --num_labels=9 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_none.json --replay=0
# mabsa_2017
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_none.json --replay=0

# masc_2017
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_and_2017_none.json --lr=1e-5 --step_size=5 --gamma=0.5 --replay=0
# mate_2017
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_and_2017_none.json --replay=0
# mner_2017
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015_and_2017_none.json --replay=0
# mabsa_2017
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1517none.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015_and_2017_none.json --replay=0


# lwf
# masc_2017
python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_lwf.json --lr=1e-5 --step_size=5 --gamma=0.5 --lwf=1
# mate_2017
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_lwf.json --lwf=1
# mner_2017
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --num_labels=9 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_lwf.json --lwf=1
# mabsa_2017
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_lwf.json --lwf=1

# masc_2017
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_and_2017_lwf.json --lr=1e-5 --step_size=5 --gamma=0.5 --lwf=1
# mate_2017
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_and_2017_lwf.json --lwf=1
# mner_2017
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015_and_2017_lwf.json --lwf=1
# mabsa_2017
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1517lwf.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015_and_2017_lwf.json --lwf=1


# si

# masc_2017
python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_mas.json --lr=1e-5 --step_size=5 --gamma=0.5 --si=1
# mate_2017
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_mas.json --si=1
# mner_2017
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --num_labels=9 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_mas.json --si=1
# mabsa_2017
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_mas.json --si=1

# masc_2017
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_and_2017_mas.json --lr=1e-5 --step_size=5 --gamma=0.5 --si=1
# mate_2017
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_and_2017_mas.json --si=1
# mner_2017
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015_and_2017_mas.json --si=1
# mabsa_2017
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1517si.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015_and_2017_mas.json --si=1


# mas

# masc_2017
python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_si.json --lr=1e-5 --step_size=5 --gamma=0.5 --mas=1
# mate_2017
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_si.json --mas=1
# mner_2017
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --num_labels=9 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_si.json --mas=1
# mabsa_2017
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_and_2017_si.json --mas=1

# masc_2017
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_and_2017_si.json --lr=1e-5 --step_size=5 --gamma=0.5 --mas=1
# mate_2017
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_and_2017_si.json --mas=1
# mner_2017
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/3.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015_and_2017_si.json --mas=1
# mabsa_2017
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1517mas.pt --output_model_path=/root/autodl-tmp/checkpoints/3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015_and_2017_si.json --mas=1

shutdown -h now