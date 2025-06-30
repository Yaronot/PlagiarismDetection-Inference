clone this to your home folder in the HUJI cluster

download best_siamese_bert.pth from release and put in the same directory in the HUJI cluster

1. **STEP 1 - Setting GPU env and GPU node access (NVIDIA L4), caching BERT model**:
   ```bash
   ~/start_gpu_session.sh"

1. **STEP 2 - INFERENCE 2 ARTICLES**:
   ```bash
   ./run_plagiarism_analysis.sh --max_articles 2"

1. **STEP 3 - SAVE RESULTS AND EXIT (csv, json, txt)**:
   ```bash
   ~/backup_and_exit.sh
   exit"













 


