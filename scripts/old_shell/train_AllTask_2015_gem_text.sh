source /d/ProgramData/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train__.txt --test_text_file=data/MASC/twitter2015/test__.txt --dev_text_file=data/MASC/twitter2015/dev__.txt --train_info_json=checkpoints/train_info_twitter2015_gem_text.json --num_labels=3 --mode=text_only --epoch=5 --lr=1e-5 --step_size=2 --gamma=0.1 --replay=0 --ewc=0 --lwf=0 --gem=1
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train__.txt --test_text_file=data/MASC/twitter2015/test__.txt --dev_text_file=data/MASC/twitter2015/dev__.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_gem_text.json --replay=0 --ewc=0 --lwf=0 --gem=1
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train__.txt --test_text_file=data/MNER/twitter2015/test__.txt --dev_text_file=data/MNER/twitter2015/dev__.txt --num_labels=9 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_gem_text.json --replay=0 --ewc=0 --lwf=0 --gem=1
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/15gem.pt --num_labels=7 --mode=text_only --train_text_file=data/MASC/twitter2015/train__.txt --test_text_file=data/MASC/twitter2015/test__.txt --dev_text_file=data/MASC/twitter2015/dev__.txt --train_info_json=checkpoints/train_info_twitter2015_gem_text.json --replay=0 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5 --ewc=0 --lwf=0 --gem=1
