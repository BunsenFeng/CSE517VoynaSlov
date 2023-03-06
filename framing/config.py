# MFC Related
KFOLD=10
PATH_TO_MFC="./data/mfc_v4.0/"
PATH_TO_DATA="./data/mfc_processed/"
PATH_TO_ANNOTATED_VOYNASLOV="./annotated/final_annotation.tsv"
ALL_TOPICS = ['climate','deathpenalty', 'guncontrol','immigration','samesex','tobacco']
DATA_OPTION = ['random', 'random_stratified'] + ALL_TOPICS
DATA_OPTION = DATA_OPTION+ [d+"_filtered" for d in DATA_OPTION]
FRAMES =   {"1":"Economic", "2": "Capacity and Resources", "3":" Morality", "4":"Fairness and Equality", "5":"Legality, Constitutionality, Jurisdiction", "6":"Policy Prescription and Evaluation", "7":"Crime and Punishment", "8":"Security and Defense", "9":"Health and Safety", "10":"Quality of Life", "11":"Cultural Identity", "12":"Public Sentiment", "13":"Political", "14":"External Regulation and Reputation", "15": "Other"}
FRAME2NUM = {frame:int(num) for num, frame in FRAMES.items()}

# Transformer-related 
PATH_TRANSFORMER_CACHE = "/usr1/home/chanyoun/transformer-models/"
TRANSFORMER_MODEL = "xlmr"
NUM_LABELS=15
MAX_LEN=80
BATCH_SIZE=64 #64 for XLM-R and 8 for XGLM
LR=5e-6

# Paths
PATH_SAVE_RUN = "/usr1/home/chanyoun/russia-ukraine/res/framing/runs/"
PATH_SAVE_SUMMARY = "/usr1/home/chanyoun/russia-ukraine/res/framing/summary/"
PATH_SAVE_MODEL = "/usr1/home/chanyoun/russia-ukraine/res/framing/models/" 