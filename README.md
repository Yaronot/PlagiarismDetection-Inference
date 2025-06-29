clone this to your home folder in the HUJI cluster

STEP 1 - Setting GPU env and GPU node access (NVIDIA L4), caching BERT model

~/start_gpu_session.sh 

STEP 2 - INFERENCE 2 ARTICLES

./run_plagiarism_analysis.sh --max_articles 2

STEP 3 - SAVE RESULTS AND EXIT (csv, json, txt)

~/backup_and_exit.sh 

exit
