# moe

# text
# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_info_json=checkpoints/train_info_twitter2015_moe_m.json --num_labels=3 --mode=text_only --lr=1e-5 --step_size=5 --gamma=0.5  --moe_adapters=1 --ddas=1
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_moe_m.json  --moe_adapters=1 --ddas=1
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_moe_m.json  --moe_adapters=1 --ddas=1
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/15moet.pt --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_twitter2015_moe_m.json  --moe_adapters=1 --ddas=1 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5

## multi
## masc_2015
#python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/15moet.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_info_json=checkpoints/train_info_twitter2015_moe_m.json --num_labels=3 --lr=1e-5 --step_size=5 --gamma=0.5  --moe_adapters=1 --ddas=1
## mate_2015
#python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2015_moe_m.json  --moe_adapters=1 --ddas=1
## mner_2015
#python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2015_moe_m.json  --moe_adapters=1 --ddas=1
## mabsa_2015
#python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/15moe.pt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2015_moe_m.json  --moe_adapters=1 --ddas=1 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5

# text
# masc_2017
python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2017_moe_m.json --lr=1e-5 --step_size=5 --gamma=0.5  --moe_adapters=1 --ddas=1
# mate_2017
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitter2017_moe_m.json  --moe_adapters=1 --ddas=1
# mner_2017
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MNER/twitter2017/train.txt --test_text_file=data/MNER/twitter2017/test.txt --dev_text_file=data/MNER/twitter2017/dev.txt --num_labels=9 --mode=text_only --train_info_json=checkpoints/train_info_twitter2017_moe_m.json  --moe_adapters=1 --ddas=1
# mabsa_2017
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/17moet.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_twitter2017_moe_m.json  --moe_adapters=1 --ddas=1
#
## multi
## masc_2017
#python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/17moet.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2017_moe_m.json --lr=1e-5 --step_size=5 --gamma=0.5  --moe_adapters=1 --ddas=1
## mate_2017
#python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitter2017_moe_m.json  --moe_adapters=1 --ddas=1
## mner_2017
#python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MNER/twitter2017/train.txt --test_text_file=data/MNER/twitter2017/test.txt --dev_text_file=data/MNER/twitter2017/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitter2017_moe_m.json  --moe_adapters=1 --ddas=1
## mabsa_2017
#python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/17moe.pt --train_text_file=data/MASC/twitter2017/train.txt --test_text_file=data/MASC/twitter2017/test.txt --dev_text_file=data/MASC/twitter2017/dev.txt --num_labels=7 --train_info_json=checkpoints/train_info_twitter2017_moe_m.json  --moe_adapters=1 --ddas=1

# text
# masc_mix
python -m scripts.train --task_name=masc --session_name=masc_1 --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitterMix_moe_m.json --lr=1e-5 --step_size=5 --gamma=0.5  --moe_adapters=1 --ddas=1
# mate_mix
python -m scripts.train --task_name=mate --session_name=mate_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --mode=text_only --train_info_json=checkpoints/train_info_twitterMix_moe_m.json  --moe_adapters=1 --ddas=1
# mner_mix
python -m scripts.train --task_name=mner --session_name=mner_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --num_labels=9 --mode=text_only --train_info_json=checkpoints/train_info_twitterMix_moe_m.json  --moe_adapters=1 --ddas=1
# mabsa_mix
python -m scripts.train --task_name=mabsa --session_name=mabsa_1 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/mixmoet.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=7 --mode=text_only --train_info_json=checkpoints/train_info_twitterMix_moe_m.json  --moe_adapters=1 --ddas=1

## multi
## masc_mix
#python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/mixmoet.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitterMix_moe_m.json --lr=1e-5 --step_size=5 --gamma=0.5  --moe_adapters=1 --ddas=1
## mate_mix
#python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=3 --train_info_json=checkpoints/train_info_twitterMix_moe_m.json  --moe_adapters=1 --ddas=1
## mner_mix
#python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/1.pt --train_text_file=data/MNER/mix/train.txt --test_text_file=data/MNER/mix/test.txt --dev_text_file=data/MNER/mix/dev.txt --num_labels=9 --train_info_json=checkpoints/train_info_twitterMix_moe_m.json  --moe_adapters=1 --ddas=1
## mabsa_mix
#python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=/root/autodl-tmp/checkpoints/1.pt --output_model_path=/root/autodl-tmp/checkpoints/mixmoe.pt --train_text_file=data/MASC/mix/train.txt --test_text_file=data/MASC/mix/test.txt --dev_text_file=data/MASC/mix/dev.txt --num_labels=7 --train_info_json=checkpoints/train_info_twitterMix_moe_m.json  --moe_adapters=1 --ddas=1
