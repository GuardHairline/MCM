# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=checkpoints/model_mnre.pt --output_model_path=checkpoints/model_mabsa.pt --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_AllDataset_none_text.json --replay=0

# masc_2017
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=checkpoints/model_mabsa.pt --output_model_path=checkpoints/model_masc_2.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --image_dir=data/MASC/twitter2017/images --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_AllDataset_none_text.json --replay=0
# mate_2017
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=checkpoints/model_masc_2.pt --output_model_path=checkpoints/model_mate_2.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --image_dir=data/MASC/twitter2017/images --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_AllDataset_none_text.json --epochs=5 --lr=1e-5 --replay=0
# mner_2017
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=checkpoints/model_mate_2.pt --output_model_path=checkpoints/model_mner_2.pt --train_text_file=data/MNER/twitter2017/train.txt --test_text_file=data/MNER/twitter2017/test.txt --dev_text_file=data/MNER/twitter2017/dev.txt --image_dir=data/MNER/twitter2017/images --num_labels=11 --mode=text_only --train_info_json=checkpoints/train_info_AllDataset_none_text.json --replay=0
# mabsa_2017
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=checkpoints/model_mner_2.pt --output_model_path=checkpoints/model_mabsa_2.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --image_dir=data/MASC/twitter2017/images --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_AllDataset_none_text.json --replay=0

# masc_mix
python -m scripts.train --task_name=masc --session_name=masc_3 --pretrained_model_path=checkpoints/model_mabsa_2.pt --output_model_path=checkpoints/model_masc_3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --image_dir=data/MASC/mix/images --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_AllDataset_none_text.json --replay=0
# mate_mix
python -m scripts.train --task_name=mate --session_name=mate_3 --pretrained_model_path=checkpoints/model_masc_3.pt --output_model_path=checkpoints/model_mate_3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --image_dir=data/MASC/mix/images --num_labels=3 --mode=text_only --epochs=5 --lr=1e-5 --train_info_json=checkpoints/train_info_AllDataset_none_text.json --replay=0
# mner_mix
python -m scripts.train --task_name=mner --session_name=mner_3 --pretrained_model_path=checkpoints/model_mate_3.pt --output_model_path=checkpoints/model_mner_3.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --image_dir=data/MNER/mix/images --num_labels=11 --mode=text_only --train_info_json=checkpoints/train_info_AllDataset_none_text.json --replay=0
# mabsa_mix
python -m scripts.train --task_name=mabsa --session_name=mabsa_3 --pretrained_model_path=checkpoints/model_mner_3.pt --output_model_path=checkpoints/model_mabsa_3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --image_dir=data/MASC/mix/images --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_AllDataset_none_text.json --replay=0
