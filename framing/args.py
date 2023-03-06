import argparse
from datetime import datetime
from config import MAX_LEN, BATCH_SIZE, LR, TRANSFORMER_MODEL, DATA_OPTION

parser = argparse.ArgumentParser(description='args for model training')

modelname2model = {"xlmr":"xlm-roberta-base","xglm":"facebook/xglm-564M","xlmrL": "xlm-roberta-large", "bert":"bert-base-cased"}
maxbatch_bymodel = {"xlmr": 64, "xlmrL": 16, "bert": 64, "xglm": 8}

parser.add_argument('--modelname', '-mn', action='store', default=TRANSFORMER_MODEL, choices=["xlmr","xglm", "xlmrL", "bert"])
parser.add_argument('--model', '-model', action='store', default=None)
parser.add_argument('--mode', '-m', action='store', default='multi', choices=["multi","ovr"])
parser.add_argument('--data', '-d', action='store', default='random_stratified', choices=DATA_OPTION)
parser.add_argument('--fewshot','-few', action='store_true', default=False, help='include 50 in-domain examples in training')

parser.add_argument('--batch_size','-bs', action='store', default=BATCH_SIZE, type=int, help='batch size')
parser.add_argument('--max_len','-ml', action='store', default=MAX_LEN, type=int, help='max token length')
parser.add_argument('--lr','-lr', action='store', default=LR, type=float, help='learning rate')
parser.add_argument('--accumulation_steps','-as', action='store', default=1, type=int, help='learning rate')

parser.add_argument('--kfold','-k', action='store', default=0, type=int, help='the idx of fold')
parser.add_argument('--foldidx','-f', action='store', default=0, type=int, help='the idx of data fold')
parser.add_argument('--epoch','-ep', action='store', default=10, type=int, help='number of epochs')
parser.add_argument('--final_epoch','-fep', action='store', default=12, type=int, help='number of epochs')
parser.add_argument('--ep_skip_eval','-ese', action='store', default=0, type=int, help='ep to skip evaluation')

parser.add_argument('--thre','-th', action='store', default=0.5, type=float)
parser.add_argument('--fscore','-fscore', action='store', default="0.5", choices=["1","2","0.5"])
parser.add_argument('--report_every', '-re', action='store', type=int, default=1000)
parser.add_argument('--num_test','-nt', action='store', default=500, type=int, help='number of samples for test')
parser.add_argument('--bool_test','-test', action='store_true', default=False, help='test (sample training exmaples to 100)')
parser.add_argument('--bool_cpu','-cpu', action='store_true', default=False, help='only use cpu')
parser.add_argument('--bool_half','-half', action='store_true', default=False, help='use half prevision version model')
parser.add_argument('--bool_no_training','-no', action='store_true', default=False, help='do not train')
parser.add_argument('--bool_final_training','-final', action='store_true', default=False, help='trian the final model to use for the analysis')
parser.add_argument('--bool_save','-s', action='store_true', default=False)
parser.add_argument('--verbose','-v', action='store_true', default=False,
                    help='verbose logging')

args = parser.parse_args()
args.model = modelname2model[args.modelname]

if args.batch_size > maxbatch_bymodel[args.modelname]:
    args.batch_size = maxbatch_bymodel[args.modelname]

if args.modelname in ["xglm", "xlmrL"]:
    args.report_every = args.report_every * (64/args.batch_size)

if args.data.endswith("_filtered"):
    args.report_every = args.report_every / 2

timestamp = datetime.now().strftime("%m%d_%H%M")
