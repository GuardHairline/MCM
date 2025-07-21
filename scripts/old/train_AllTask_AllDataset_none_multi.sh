source /d/ProgramData/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
# masc_2015
#python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=checkpoints/masc_15_m.pt --train_info_json=checkpoints/train_info_masc_none_multi.json --num_labels=3 --lr=1e-5 --step_size=5 --gamma=0.5 --replay=0 --fusion_strategy=concat
# mate_2015
#python -m scripts.train --task_name=mate --session_name=mate_1 --output_model_path=checkpoints/mate_15_m.pt --num_labels=3 --train_info_json=checkpoints/train_info_mate_none_multi.json --replay=0 --fusion_strategy=concat
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_1 --output_model_path=checkpoints/mner_15_m.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_mner_none_multi.json --replay=0 --fusion_strategy=concat
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --output_model_path=checkpoints/mabsa_15_m.pt --num_labels=7 --train_info_json=checkpoints/train_info_mabsa_none_multi.json --replay=0 --fusion_strategy=concat --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5
## mnre_2015
#python -m scripts.train --task_name=mnre --session_name=mnre_1 --pretrained_model_path=checkpoints/model_mabsa_15_n.pt --output_model_path=checkpoints/model_mnre_15_n.pt --train_text_file=data/MNRE/twitter2015/train.txt --test_text_file=data/MNRE/twitter2015/test.txt --dev_text_file=data/MNRE/twitter2015/dev.txt --num_labels=23 --train_info_json=checkpoints/train_info_twitter2015_none_multi.json --replay=0 --fusion_strategy=concat
#
## masc_2017
#python -m scripts.train --task_name=masc --session_name=masc_2 --output_model_path=checkpoints/model_masc_17_n.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2017_none_multi.json --lr=1e-5 --step_size=5 --gamma=0.5 --replay=0 --fusion_strategy=concat
## mate_2017
#python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=checkpoints/model_masc_17_n.pt --output_model_path=checkpoints/model_mate_17_n.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2017_none_multi.json --replay=0 --fusion_strategy=concat
## mner_2017
#python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=checkpoints/model_mate_17_n.pt --output_model_path=checkpoints/model_mner_17_n.pt --train_text_file=data/MNER/twitter2017/train.txt --test_text_file=data/MNER/twitter2017/test.txt --dev_text_file=data/MNER/twitter2017/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2017_none_multi.json --replay=0 --fusion_strategy=concat
## mabsa_2017
#python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=checkpoints/model_mner_17_n.pt --output_model_path=checkpoints/model_mabsa_17_n.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2017_none_multi.json --replay=0 --fusion_strategy=concat
## mnre_2017
#python -m scripts.train --task_name=mnre --session_name=mnre_2 --pretrained_model_path=checkpoints/model_mabsa_17_n.pt --output_model_path=checkpoints/model_mnre_17_n.pt --train_text_file=data/MNRE/twitter2017/train.txt --test_text_file=data/MNRE/twitter2017/test.txt --dev_text_file=data/MNRE/twitter2017/dev.txt --num_labels=23 --train_info_json=checkpoints/train_info_twitter2017_none_multi.json --replay=0 --fusion_strategy=concat
#
## masc_mix
#python -m scripts.train --task_name=masc --session_name=masc_3 --output_model_path=checkpoints/model_masc_m_n.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitterMix_none_multi.json --lr=1e-5 --step_size=5 --gamma=0.5 --replay=0 --fusion_strategy=concat
## mate_mix
#python -m scripts.train --task_name=mate --session_name=mate_3 --pretrained_model_path=checkpoints/model_masc_m_n.pt --output_model_path=checkpoints/model_mate_m_n.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitterMix_none_multi.json --replay=0 --fusion_strategy=concat
## mner_mix
#python -m scripts.train --task_name=mner --session_name=mner_3 --pretrained_model_path=checkpoints/model_mate_m_n.pt --output_model_path=checkpoints/model_mner_m_n.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitterMix_none_multi.json --replay=0 --fusion_strategy=concat
## mabsa_mix
#python -m scripts.train --task_name=mabsa --session_name=mabsa_3 --pretrained_model_path=checkpoints/model_mner_m_n.pt --output_model_path=checkpoints/model_mabsa_m_n.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=7 --train_info_json=checkpoints/train_info_twitterMix_none_multi.json --replay=0 --fusion_strategy=concat
## mnre
#python -m scripts.train --task_name=mnre --session_name=mnre_3 --pretrained_model_path=checkpoints/model_mabsa_m_n.pt --output_model_path=checkpoints/model_mnre_m_n.pt --train_text_file=data/MNRE/mix/train.txt --test_text_file=data/MNRE/mix/test.txt --dev_text_file=data/MNRE/mix/dev.txt --num_labels=23 --train_info_json=checkpoints/train_info_twitterMix_none_multi.json --replay=0 --fusion_strategy=concat
