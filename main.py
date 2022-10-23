import argparse
import os
from datetime import datetime

import torch.optim
from tensorboardX import SummaryWriter
from data import DataIter
from ptrmodel import STN
from ptr import train
from utils import save_model, load_model

# Parse arguments
parser = argparse.ArgumentParser(description='product short title')
parser.add_argument('--proceed', type=bool, default=False, help='proceed learning, or start new one learning process')

parser.add_argument('--cuda', type=bool, default=True, help='whether to use cuda')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--emb_size', type=int, default=128, help='vocabulary size')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden size of model')
parser.add_argument('--n_layers', type=int, default=1, help='number of layers')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

parser.add_argument('--clip', type=float, default=1, help='gradient clipping threshold')
parser.add_argument('--logdir', type=str, default='./logs/', help='log directory')
parser.add_argument('--model_name', type=str, default='model.pkl', help='name of model file')

parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')



args = parser.parse_args()

models_folder = './model/'
model_filename = os.path.join(models_folder, args.model_name)
os.makedirs(os.path.dirname(model_filename), exist_ok=True)
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if args.proceed:
    print('Proceed learning of exciting model')
    assert os.path.exists(model_filename), "Can't find pretrained model file"
    model, optimizer, last_iter, args, log_folder = load_model(model_filename)
else:
    log_folder = os.path.join(args.logdir, datetime.today().strftime("%Y_%m_%d-%H_%M_%S"))
    last_iter = 0

writer = SummaryWriter(log_folder)

device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

print('Loading dataset...')
train_iterator,test_iterator,vocab_size,ner_vocab,criterion= DataIter(args.batch_size,device)
print("vocab_size:{}".format(vocab_size))

if not args.proceed:
    model = STN(vocab_size,ner_vocab, args.emb_size,args.hidden_size, args.n_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
        



num_iters = args.epochs
for i in range(last_iter, num_iters):
    try:
        loss, grads = train(model,train_iterator,optimizer,criterion, args.clip)
        writer.add_scalar('PGNN/loss', loss, i)
        writer.add_scalar('PGNN/grad', grads, i)
        print('[{:.2f}%] Iteration {};\t Loss: {:.5f}\t Gradients: {:.5f}'.format((i+1) / num_iters * 100, i, loss,
                                                                                      grads))
           
    except KeyboardInterrupt:
        last_iter = i
        break

save_model(model_filename, model, optimizer, last_iter, args, log_folder)

