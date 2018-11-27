source activate summer
python3 stage1.py
python2 stage2.py
python3 stage3.py
matlab -nodesktop -nosplash -r "generate_lpq_features.m"
# python3 stage4.py
python2 stage5.py
