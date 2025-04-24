import os

from datetime import datetime
import pickle

import numpy as np
from sacred import Experiment
from sacred.commands import print_config, save_config
from sacred.observers import FileStorageObserver
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import *
ex = Experiment('train_original')

# Constants
logging_freq = 100
saving_freq = 200

@ex.config
def config():
    root = 'runs'
    device = 'cuda:0'
    log = True
    w_size = 31
    spec = 'Mel'
    resume_iteration = None
    train_on = 'MAPS'
    n_heads = 4
    position = True
    iteration = 10
    VAT_start = 0
    alpha = 1
    VAT = True
    XI = 1e-4
    eps = 0.5
    small = False
    KL_Div = False
    reconstruction = False

    batch_size = 8
    train_batch_size = 1
    sequence_length = 16384
    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    epoches = 20000
    step_size_up = 100
    max_lr = 1e-4
    learning_rate = 1e-3
    learning_rate_decay_steps = 1000
    learning_rate_decay_rate = 0.98

    leave_one_out = None
    clip_gradient_norm = 3
    validation_length = sequence_length
    refresh = False

    logdir = f'{root}/Unet-recons={reconstruction}-XI={XI}-eps={eps}-alpha={alpha}-train_on=small_{small}_{train_on}-w_size={w_size}-n_heads={n_heads}-lr={learning_rate}-'+ datetime.now().strftime('%y%m%d-%H%M%S')
    ex.observers.append(FileStorageObserver.create(logdir))

@ex.automain
def train(spec, resume_iteration, train_on, batch_size, sequence_length,w_size, n_heads, small, train_batch_size,
          learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out, position, alpha, KL_Div,
          clip_gradient_norm, validation_length, refresh, device, epoches, logdir, log, iteration, VAT_start, VAT, XI, eps,
          reconstruction): 
    print_config(ex.current_run)

    dataset_path = '/content/pMaps'  # Adjusted path for Colab zip extract
    groups = ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'ENSTDkCl', 'SptkBGAm']

    supervised_set, unsupervised_set, validation_dataset, full_validation = prepare_VAT_dataset(
        path=dataset_path,
        groups=groups,
        sequence_length=sequence_length,
        validation_length=sequence_length,
        refresh=refresh,
        device=device,
        small=small,
        supersmall=True,
        dataset=train_on)  

    if VAT:
        unsupervised_loader = DataLoader(unsupervised_set, batch_size, shuffle=True, drop_last=True)

    val_batch_size = min(len(validation_dataset), 4)
    supervised_loader = DataLoader(supervised_set, train_batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(validation_dataset, val_batch_size, shuffle=False, drop_last=True)
    batch_visualize = next(iter(valloader))

    ds_ksize, ds_stride = (2,2),(2,2)
    if resume_iteration is None:
        model = UNet(ds_ksize, ds_stride, log=log, reconstruction=reconstruction,
                     mode='imagewise', spec=spec, device=device, XI=XI, eps=eps)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        trained_dir = 'trained_MAPS'
        model_path = os.path.join(trained_dir, f'{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(trained_dir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    for ep in range(1, epoches + 1):
        if VAT:
            predictions, losses, optimizer = train_VAT_model(model, iteration, ep, supervised_loader, unsupervised_loader,
                                                             optimizer, scheduler, clip_gradient_norm, alpha, VAT, VAT_start)
        else:
            predictions, losses, optimizer = train_VAT_model(model, iteration, ep, supervised_loader, None,
                                                             optimizer, scheduler, clip_gradient_norm, alpha, VAT, VAT_start)
        loss = sum(losses.values())

        if ep == 1:
            writer = SummaryWriter(logdir)

        tensorboard_log(batch_visualize, model, validation_dataset, supervised_loader,
                        ep, logging_freq, saving_freq, n_heads, logdir, w_size, writer,
                        VAT and ep >= VAT_start, VAT_start, reconstruction)

        if ep % saving_freq == 0:
            torch.save(model.state_dict(), os.path.join(logdir, f'model-{ep}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))

        for key, value in losses.items():
            writer.add_scalar(key, value.item(), global_step=ep)

    print('Training finished, now evaluating on the MAPS test split (full songs)')
    with torch.no_grad():
        model.eval()
        metrics = evaluate_wo_velocity(tqdm(full_validation), model, reconstruction=False,
                                       save_path=os.path.join(logdir,'./MIDI_results'))

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')

    export_path = os.path.join(logdir, 'result_dict')
    pickle.dump(metrics, open(export_path, 'wb'))
