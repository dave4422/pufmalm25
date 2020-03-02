"""
Reproduce Model-agnostic Meta-learning results (supervised only) of Finn et al
"""
from torch.utils.data import DataLoader
from torch import nn
import argparse

from few_shot.datasets import RoPUF
from few_shot.core import NShotTaskSampler, create_nshot_task_label, EvaluateFewShot
from few_shot.maml import meta_gradient_step
from few_shot.models import FewShotClassifierPUF
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH
from few_shot.metrics import categorical_accuracy


setup_dirs()
assert torch.cuda.is_available()
#TODO change to cuda
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--n', default=1, type=int)
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--q', default=1, type=int)  # Number of examples per class to calculate meta gradients with
parser.add_argument('--inner-train-steps', default=1, type=int)
parser.add_argument('--inner-val-steps', default=3, type=int)
parser.add_argument('--inner-lr', default=0.4, type=float)
parser.add_argument('--meta-lr', default=0.001, type=float)
parser.add_argument('--meta-batch-size', default=32, type=int)
parser.add_argument('--order', default=1, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--epoch-len', default=100, type=int)
parser.add_argument('--eval-batches', default=20, type=int)
parser.add_argument('--test-board', default='D070802_64', type=str)
parser.add_argument('--challenge-size', default=64, type=int)
parser.add_argument("--load-indexed", type=str2bool, nargs='?',const=True, default=False)
#parser.add_argument('--boards-per-task', default=25, type=int)



args = parser.parse_args()
challenge_size = args.challenge_size
test_board = args.test_board
load_indexed = args.load_indexed


if args.dataset == 'roPUF':
    dataset_class = RoPUF
else:
    raise(ValueError('Unsupported dataset'))

param_str = f'{args.dataset}_order={args.order}_n={args.n}_k={args.k}_metabatch={args.meta_batch_size}_' \
            f'train_steps={args.inner_train_steps}_val_steps={args.inner_val_steps}'
print(param_str)


###################
# Create datasets #
###################
background = dataset_class('background',challenge_size,test_board,load_indexed)
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, args.epoch_len, n=args.n, k=args.k, q=args.q,
                                   num_tasks=args.meta_batch_size),
    num_workers=8
)
evaluation = dataset_class('evaluation',challenge_size,test_board,load_indexed)
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, args.eval_batches, n=args.n, k=args.k, q=args.q,
                                   num_tasks=1),
    num_workers=8
)

test_untouched = dataset_class('test_untouched',challenge_size,test_board,load_indexed)
test_taskloader = DataLoader(
    test_untouched,
    batch_sampler=NShotTaskSampler(evaluation, 800, n=args.n, k=args.k, q=args.q,
                                   num_tasks=1),
    num_workers=8
)


############
# Training #
############
print(f'Training MAML on {args.dataset}...')
meta_model = FewShotClassifierPUF(in_features = challenge_size).to(device, dtype=torch.double)
meta_optimiser = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
# TODO change to nn.BCELoss().to(device)  ??
loss_fn =  nn.BCELoss().to(device)#nn.CrossEntropyLoss().to(device)


def prepare_meta_batch(n, k, q, meta_batch_size):
    def prepare_meta_batch_(batch):
        x, y = batch

        #print(batch.dim)
        #print(x.shape)
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        # evaluate the fast model on and generate meta-gradients
        #print(x.shape)
        #TODO is this correct?
        x = x.reshape(meta_batch_size, n + q, x.shape[-1])
        # Move to device
        x = x.double().to(device)

        #print(x)
        y = y.reshape(meta_batch_size, n + q, 1)
        y = y.double().to(device)
        #print(x.shape)
        #print(y.reshape(meta_batch_size, n*k + q*k, 1).shape)
        # Create label
        #y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
        return x, y

    return prepare_meta_batch_





callbacks = [
    EvaluateFewShot(
        eval_fn=meta_gradient_step,
        num_tasks=args.eval_batches,
        n_shot=args.n,
        k_way=args.k,
        q_queries=args.q,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_meta_batch(args.n, args.k, args.q, 1), #changed meta_batch_size to 1 bc of line83 vs l 90
        # MAML kwargs
        inner_train_steps=args.inner_val_steps,
        inner_lr=args.inner_lr,
        device=device,
        order=args.order,
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/maml2/{param_str}.pth',
        monitor=f'val_{args.n}-shot_{args.k}-way_acc'
    ),
    ReduceLROnPlateau(patience=10, factor=0.5, monitor=f'val_loss'),
    CSVLogger(PATH + f'/logs/maml2/{param_str}.csv'),
]



fit(
    meta_model,
    meta_optimiser,
    loss_fn,
    epochs=args.epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=meta_gradient_step,
    fit_function_kwargs={'n_shot': args.n, 'k_way': args.k, 'q_queries': args.q,
                         'train': True,
                         'order': args.order, 'device': device, 'inner_train_steps': args.inner_train_steps,
                         'inner_lr': args.inner_lr},
)

seen = 0
totals = {'loss': 0, 'ca': 0}
print(test_taskloader.dataset.subset)
for batch_index, batch in enumerate(test_taskloader):
    x, y = (prepare_meta_batch(args.n, args.k, args.q, 1))(batch)

    loss, y_pred = meta_gradient_step(
        meta_model,
        meta_optimiser,
        loss_fn,
        x,
        y,
        n_shot=args.n,
        k_way=args.k,
        q_queries=args.q,
        train=False,
        inner_train_steps=args.inner_val_steps,
        inner_lr=args.inner_lr,
        device=device,
        order=args.order,
    )

    seen += y_pred.shape[0]

    totals['loss'] += loss.item() * y_pred.shape[0]
    totals['ca'] += categorical_accuracy(y[:,-1:,:], y_pred)# * y_pred.shape[0]

print(totals['loss'] / seen)
print(totals['ca'] / seen)
