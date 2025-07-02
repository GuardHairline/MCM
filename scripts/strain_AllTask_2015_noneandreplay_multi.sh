#none

# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_info_json=checkpoints/train_info_twitter2015_none_m.json --num_labels=3 --lr=1e-5 --step_size=5 --gamma=0.5 --replay=0
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5

# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_info_json=checkpoints/train_info_twitter2015_none_m.json --num_labels=3 --lr=1e-5 --step_size=5 --gamma=0.5 --replay=0
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5


# replay
# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_info_json=checkpoints/train_info_twitter2015_none_m.json --num_labels=3 --lr=1e-5 --step_size=5 --gamma=0.5 --replay=0
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5

# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_info_json=checkpoints/train_info_twitter2015_none_m.json --num_labels=3 --lr=1e-5 --step_size=5 --gamma=0.5 --replay=0
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015_none_m.json --replay=0 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5
