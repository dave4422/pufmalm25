import torch
from config import PATH
from torch.utils.data import DataLoader
from few_shot.models import FewShotClassifierPUF
from few_shot.datasets import RoPUF
from few_shot.core import NShotTaskSampler, create_nshot_task_label, EvaluateFewShot
from few_shot.maml import meta_gradient_step
from torch import nn
challenge_size = 64
meta_batch_size = 70

n = 70
k = 1
q = 1

test_board = 'D080157'
eval_batches = 300
param_str = 'roPUF_order=2_n=70_k=1_metabatch=70_train_steps=40_val_steps=10'


assert torch.cuda.is_available()
#TODO change to cuda
device = torch.device('cuda')





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
        x = x.float().to(device)

        #print(x)
        y = y.reshape(meta_batch_size, n + q, 1)
        y = y.float().to(device)
        #print(x.shape)
        #print(y.reshape(meta_batch_size, n*k + q*k, 1).shape)
        # Create label
        #y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
        return x, y

    return prepare_meta_batch_

data = RoPUF('test_untouched',challenge_size,test_board,False)
data_taskloader = DataLoader(
    data,
    batch_sampler=NShotTaskSampler(data, eval_batches, n=n, k=k, q=q,
                                   num_tasks=1),
    num_workers=8
)



model = FewShotClassifierPUF(in_features = challenge_size)
model.load_state_dict(torch.load(PATH + f'/models/maml2/{param_str}.pth'))
model.eval()
model.to(device)


meta_optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
# TODO change to nn.BCELoss().to(device)  ??
loss_fn =  nn.BCELoss().to(device)
sum = 0
for batch_index, batch in enumerate(data_taskloader):
    x, y = prepare_meta_batch(n,k,q,1)(batch)

    loss, y_pred = meta_gradient_step(
        model,
        meta_optimiser,
        loss_fn,
        x,
        y,
        n_shot=n,
        k_way=k,
        q_queries=q,
        train=False,
        order=2,
        inner_train_steps=40,
        inner_lr=0.01,
        device=device,
    )

    #do something
    #print(y_pred)
    #print(y)
    y = y[:,-q:,:]

    sum += torch.eq(y_pred, torch.flatten(y)).sum().item() / y_pred.shape[0]
    #print('here')
#print('here2')
print(sum/eval_batches)
#print('here3')
