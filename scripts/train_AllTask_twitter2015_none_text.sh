# masc
python -m scripts.train --task_name=masc --num_labels=3 --session_name=masc_1 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_none_text.json --replay=0
# mate
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=checkpoints/model_1.pt --output_model_path=checkpoints/model_mate.pt --num_labels=3 --epochs=5 --lr=1e-5 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_none_text.json --replay=0
# mner
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=checkpoints/model_mate.pt --output_model_path=checkpoints/model_mner.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --image_dir=data/MNER/twitter2015/images --num_labels=9 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_none_text.json --replay=0
# mnre
python -m scripts.train --task_name=mnre --session_name=mnre_1 --pretrained_model_path=checkpoints/model_mner.pt --output_model_path=checkpoints/model_mnre.pt --train_text_file=data/MNRE/mix/mnre_txt/train.txt --test_text_file=data/MNRE/mix/mnre_txt/test.txt --dev_text_file=data/MNRE/mix/mnre_txt/dev.txt --image_dir=data/MNRE/mix/mnre_image --num_labels=23 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_none_text.json --replay=0
# mabsa
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=checkpoints/model_mnre.pt --output_model_path=checkpoints/model_mabsa.pt --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_none_text.json --replay=0