source /d/ProgramData/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

# session_name pretrained_model_path output_model_path train_info_json mode


# mas

# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=checkpoints/final3/15mas.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --train_info_json=checkpoints/final3/train_info_twitter2015_mas_text.json --num_labels=3 --mode=multimodal --epoch=5 --lr=1e-5 --step_size=2 --gamma=0.1 --replay=0 --ewc=0 --lwf=0 --si=0 --mas=1
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --num_labels=3 --mode=multimodal --train_info_json=checkpoints/final3/train_info_twitter2015_mas_text.json --replay=0 --ewc=0 --lwf=0 --si=0 --mas=1
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 --mode=multimodal --train_info_json=checkpoints/final3/train_info_twitter2015_mas_text.json --replay=0 --ewc=0 --lwf=0 --si=0 --mas=1
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/final3/15masm.pt --num_labels=7 --mode=multimodal --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --train_info_json=checkpoints/final3/train_info_twitter2015_mas_text.json --replay=0 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5 --ewc=0 --lwf=0 --si=0 --mas=1


# si

# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=checkpoints/final3/15si.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --train_info_json=checkpoints/final3/train_info_twitter2015_si_text.json --num_labels=3 --mode=multimodal --epoch=5 --lr=1e-5 --step_size=2 --gamma=0.1 --replay=0 --ewc=0 --lwf=0 --si=1
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --num_labels=3 --mode=multimodal --train_info_json=checkpoints/final3/train_info_twitter2015_si_text.json --replay=0 --ewc=0 --lwf=0 --si=1
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 --mode=multimodal --train_info_json=checkpoints/final3/train_info_twitter2015_si_text.json --replay=0 --ewc=0 --lwf=0 --si=1
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/final3/15sim.pt --num_labels=7 --mode=multimodal --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --train_info_json=checkpoints/final3/train_info_twitter2015_si_text.json --replay=0 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5 --ewc=0 --lwf=0 --si=1

# lwf

# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=checkpoints/final3/15mas.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --train_info_json=checkpoints/final3/train_info_twitter2015_lwf_text.json --num_labels=3 -mode=multimodal --epoch=5 --lr=1e-5 --step_size=2 --gamma=0.1 --replay=0 --ewc=0 --lwf=1
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --num_labels=3 -mode=multimodal --train_info_json=checkpoints/final3/train_info_twitter2015_lwf_text.json --replay=0 --ewc=0 --lwf=1
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 -mode=multimodal --train_info_json=checkpoints/final3/train_info_twitter2015_lwf_text.json --replay=0 --ewc=0 --lwf=1
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/final3/15masm.pt --num_labels=7 -mode=multimodal --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --train_info_json=checkpoints/final3/train_info_twitter2015_lwf_text.json --replay=0 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5 --ewc=0 --lwf=1

#ewc
# masc_2015
python -m scripts.train --task_name=masc --session_name=masc_2 --pretrained_model_path=checkpoints/final3/15ewc.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --train_info_json=checkpoints/final3/train_info_twitter2015_Ewc_text_0.01.json --num_labels=3 -mode=multimodal --epoch=5 --lr=1e-5 --step_size=2 --gamma=0.1 --replay=0 --ewc=1  --ewc_dir=/checkpoints/final3/ewc_params
# mate_2015
python -m scripts.train --task_name=mate --session_name=mate_2 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --num_labels=3 -mode=multimodal --train_info_json=checkpoints/final3/train_info_twitter2015_Ewc_text_0.01.json --replay=0 --ewc=1  --ewc_dir=/checkpoints/final3/ewc_params
# mner_2015
python -m scripts.train --task_name=mner --session_name=mner_2 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/1.pt --train_text_file=data/MNER/twitter2015/train.txt --test_text_file=data/MNER/twitter2015/test.txt --dev_text_file=data/MNER/twitter2015/dev.txt --num_labels=9 -mode=multimodal --train_info_json=checkpoints/final3/train_info_twitter2015_Ewc_text_0.01.json --replay=0 --ewc=1  --ewc_dir=/checkpoints/final3/ewc_params
# mabsa_2015
python -m scripts.train --task_name=mabsa --session_name=mabsa_2 --pretrained_model_path=checkpoints/1.pt --output_model_path=checkpoints/final3/15ewcm.pt --num_labels=7 -mode=multimodal --train_text_file=data/MASC/twitter2015/train.txt --test_text_file=data/MASC/twitter2015/test.txt --dev_text_file=data/MASC/twitter2015/dev.txt --train_info_json=checkpoints/final3/train_info_twitter2015_Ewc_text_0.01.json --replay=0 --epoch=20 --lr=5e-5 --step_size=10 --gamma=0.5 --ewc=1  --ewc_dir=/checkpoints/final3/ewc_params

