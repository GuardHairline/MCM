# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=checkpoints/model_masc.pt --train_info_json=checkpoints/train_info_twitter2015_replay_text.json --num_labels=3 --mode=text_only --replay=1
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=checkpoints/model_masc.pt --output_model_path=checkpoints/model_mate.pt --num_labels=3 --epochs=5 --lr=1e-5 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_replay_text.json --replay=1
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=checkpoints/model_mate.pt --output_model_path=checkpoints/model_mner.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --image_dir=data/MNER/twitter2015/images --num_labels=9 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_replay_text.json --replay=1
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=checkpoints/model_mner.pt --output_model_path=checkpoints/model_mabsa.pt --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_replay_text.json --replay=1  --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5

# masc_2017
python -m scripts.train --task_name=masc --session_name=masc_2 --output_model_path=checkpoints/model_masc_2.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --image_dir=data/MASC/twitter2017/images --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2017_replay_text.json --replay=1
# mate_2017
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=checkpoints/model_masc_2.pt --output_model_path=checkpoints/model_mate_2.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --image_dir=data/MASC/twitter2017/images --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2017_replay_text.json --epochs=5 --lr=1e-5 --replay=1
# mner_2017
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=checkpoints/model_mate_2.pt --output_model_path=checkpoints/model_mner_2.pt --train_text_file=data/MNER/twitter2017/train.txt --test_text_file=data/MNER/twitter2017/test.txt --dev_text_file=data/MNER/twitter2017/dev.txt --image_dir=data/MNER/twitter2017/images --num_labels=11 --mode=text_only --train_info_json=checkpoints/train_info_twitter2017_replay_text.json --replay=1
# mabsa_2017
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=checkpoints/model_mner_2.pt --output_model_path=checkpoints/model_mabsa_2.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --image_dir=data/MASC/twitter2017/images --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_twitter2017_replay_text.json --replay=1  --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5

# masc_mix
python -m scripts.train --task_name=masc --session_name=masc_3 --output_model_path=checkpoints/model_masc_3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --image_dir=data/MASC/mix/images --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitterMix_replay_text.json --replay=1
# mate_mix
python -m scripts.train --task_name=mate --session_name=mate_3 --pretrained_model_path=checkpoints/model_masc_3.pt --output_model_path=checkpoints/model_mate_3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --image_dir=data/MASC/mix/images --num_labels=3 --mode=text_only --epochs=5 --lr=1e-5 --train_info_json=checkpoints/train_info_twitterMix_replay_text.json --replay=1
# mner_mix
python -m scripts.train --task_name=mner --session_name=mner_3 --pretrained_model_path=checkpoints/model_mate_3.pt --output_model_path=checkpoints/model_mner_3.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --image_dir=data/MNER/mix/images --num_labels=11 --mode=text_only --train_info_json=checkpoints/train_info_twitterMix_replay_text.json --replay=1
# mabsa_mix
python -m scripts.train --task_name=mabsa --session_name=mabsa_3 --pretrained_model_path=checkpoints/model_mner_3.pt --output_model_path=checkpoints/model_mabsa_3.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --image_dir=data/MASC/mix/images --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_twitterMix_replay_text.json --replay=1  --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5
# mnre
python -m scripts.train --task_name=mnre --session_name=mnre_1 --pretrained_model_path=checkpoints/model_mabsa_3.pt --output_model_path=checkpoints/model_mnre_3.pt --train_text_file=data/MNRE/mix/mnre_txt/train.txt --test_text_file=data/MNRE/mix/mnre_txt/test.txt --dev_text_file=data/MNRE/mix/mnre_txt/dev.txt --image_dir=data/MNRE/mix/mnre_image --num_labels=23 --mode=text_only --train_info_json=checkpoints/train_info_twitterMix_replay_text.json --replay=1
