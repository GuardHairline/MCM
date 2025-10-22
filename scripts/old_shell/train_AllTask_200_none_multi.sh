source /d/ProgramData/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=checkpoints/3.pt --train_text_file=data/MASC/twitter2015/train__.txt --test_text_file=data/MASC/twitter2015/test__.txt --dev_text_file=data/MASC/twitter2015/dev__.txt --train_info_json=checkpoints/train_info_twitter2015-200_none_multi.json --num_labels=3 --epoch=5 --lr=1e-5 --step_size=2 --gamma=0.1 --replay=0
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=checkpoints/3.pt --output_model_path=checkpoints/3.pt --train_text_file=data/MASC/twitter2015/train__.txt --test_text_file=data/MASC/twitter2015/test__.txt --dev_text_file=data/MASC/twitter2015/dev__.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015-200_none_multi.json --replay=0
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=checkpoints/3.pt --output_model_path=checkpoints/3.pt --train_text_file=data/MNER/twitter2015/train__.txt --test_text_file=data/MNER/twitter2015/test__.txt --dev_text_file=data/MNER/twitter2015/dev__.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015-200_none_multi.json --replay=0
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=checkpoints/3.pt --output_model_path=checkpoints/3.pt --train_text_file=data/MASC/twitter2015/train__.txt --test_text_file=data/MASC/twitter2015/test__.txt --dev_text_file=data/MASC/twitter2015/dev__.txt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015-200_none_multi.json --replay=0 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5
# mnre_2015
python -m scripts.train --task_name=mnre --session_name=mnre_1 --pretrained_model_path=checkpoints/3.pt --output_model_path=checkpoints/3.pt --train_text_file=data/MNRE/twitter2015/train__.txt --test_text_file=data/MNRE/twitter2015/test__.txt --dev_text_file=data/MNRE/twitter2015/dev__.txt --num_labels=23 --train_info_json=checkpoints/train_info_twitter2015-200_none_multi.json --replay=0

## masc_2017
#python -m scripts.train --task_name=masc --session_name=masc_2 --output_model_path=checkpoints/2.pt --train_text_file=data/MASC/twitter2017/train__.txt --test_text_file=data/MASC/twitter2017/test__.txt --dev_text_file=data/MASC/twitter2017/dev__.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2017_none_text.json --epoch=5 --lr=1e-5 --step_size=2 --gamma=0.1 --replay=0
## mate_2017
#python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=checkpoints/2.pt --output_model_path=checkpoints/2.pt --train_text_file=data/MASC/twitter2017/train__.txt --test_text_file=data/MASC/twitter2017/test__.txt --dev_text_file=data/MASC/twitter2017/dev__.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2017_none_text.json --replay=0
## mner_2017
#python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=checkpoints/2.pt --output_model_path=checkpoints/2.pt --train_text_file=data/MNER/twitter2017/train__.txt --test_text_file=data/MNER/twitter2017/test__.txt --dev_text_file=data/MNER/twitter2017/dev__.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2017_none_text.json --replay=0
## mabsa_2017
#python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=checkpoints/2.pt --output_model_path=checkpoints/2.pt --train_text_file=data/MASC/twitter2017/train__.txt --test_text_file=data/MASC/twitter2017/test__.txt --dev_text_file=data/MASC/twitter2017/dev__.txt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2017_none_text.json --replay=0
## mnre_2017
#python -m scripts.train --task_name=mnre --session_name=mnre_2 --pretrained_model_path=checkpoints/2.pt --output_model_path=checkpoints/2.pt --train_text_file=data/MNRE/twitter2017/train__.txt --test_text_file=data/MNRE/twitter2017/test__.txt --dev_text_file=data/MNRE/twitter2017/dev__.txt --num_labels=23 --train_info_json=checkpoints/train_info_twitter2017_none_text.json --replay=0
