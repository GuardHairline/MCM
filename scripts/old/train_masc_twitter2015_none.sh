export PYTHONIOENCODING=utf-8
source /d/ProgramData/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
python -m scripts.train --task_name=masc --num_labels=3 --session_name=masc_1 --mode=text_only