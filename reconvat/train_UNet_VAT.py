import os
import pickle
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import *  # Make sure model/__init__.py exposes necessary functions/classes


def get_config():
    return {
        "device": "cuda:0",
        "spec": "Mel",
        "resume_iteration": None,
        "train_on": "MAPS",
        "sequence_length": 327680,
        "train_batch_size": 1,
        "batch_size": 8,
        "refresh": False,
        "small": True,
        "VAT": True,
        "VAT_start": 0,
        "alpha": 1,
        "XI": 1e-6,
        "eps": 2,
        "w_size": 31,
        "n_heads": 4,
        "log": True,
        "logdir": f"runs/manual_run_{datetime.now().strftime('%y%m%d-%H%M%S')}",
        "epoches": 20000,
        "learning_rate": 1e-3,
        "learning_rate_decay_steps": 1000,
        "learning_rate_decay_rate": 0.98,
        "clip_gradient_norm": 3,
        "validation_length": 327680,
        "leave_one_out": None,
        "position": True,
        "reconstruction": True,
        "iteration": 10,
        "KL_Div": False,
    }


def train(**kwargs):
    os.makedirs(kwargs['logdir'], exist_ok=True)

    print("[INFO] Loading datasets...")
    sup_set, unsup_set, val_set, full_val = prepare_VAT_dataset(
        sequence_length=kwargs['sequence_length'],
        validation_length=kwargs['validation_length'],
        refresh=kwargs['refresh'],
        device=kwargs['device'],
        small=kwargs['small'],
        supersmall=True,
        dataset=kwargs['train_on']
    )

    sup_loader = DataLoader(sup_set, kwargs['train_batch_size'], shuffle=True, drop_last=True)
    unsup_loader = DataLoader(unsup_set, kwargs['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, min(len(val_set), 4), shuffle=False, drop_last=True)
    batch_visualize = next(iter(val_loader))

    print("[INFO] Initializing model...")
    if kwargs['resume_iteration'] is None:
        model = UNet((2, 2), (2, 2), log=kwargs['log'], reconstruction=kwargs['reconstruction'],
                     mode='imagewise', spec=kwargs['spec'], device=kwargs['device'],
                     XI=kwargs['XI'], eps=kwargs['eps'])
        model.to(kwargs['device'])
        optimizer = torch.optim.Adam(model.parameters(), kwargs['learning_rate'])
        start_ep = 1
    else:
        model_path = os.path.join('trained_MAPS', f"{kwargs['resume_iteration']}.pt")
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), kwargs['learning_rate'])
        optimizer.load_state_dict(torch.load(os.path.join('trained_MAPS', 'last-optimizer-state.pt')))
        start_ep = kwargs['resume_iteration'] + 1

    scheduler = StepLR(optimizer, step_size=kwargs['learning_rate_decay_steps'], gamma=kwargs['learning_rate_decay_rate'])
    writer = SummaryWriter(kwargs['logdir'])

    print("[INFO] Starting training...")
    for ep in range(start_ep, kwargs['epoches'] + 1):
        if kwargs['VAT']:
            preds, losses, optimizer = train_VAT_model(model, kwargs['iteration'], ep, sup_loader, unsup_loader,
                                                        optimizer, scheduler, kwargs['clip_gradient_norm'],
                                                        kwargs['alpha'], kwargs['VAT'], kwargs['VAT_start'])
        else:
            preds, losses, optimizer = train_VAT_model(model, kwargs['iteration'], ep, sup_loader, None,
                                                        optimizer, scheduler, kwargs['clip_gradient_norm'],
                                                        kwargs['alpha'], kwargs['VAT'], kwargs['VAT_start'])

        if ep < kwargs['VAT_start'] or not kwargs['VAT']:
            tensorboard_log(batch_visualize, model, val_set, sup_loader, ep, 100, 200, kwargs['n_heads'],
                            kwargs['logdir'], kwargs['w_size'], writer, False, kwargs['VAT_start'], kwargs['reconstruction'])
        else:
            tensorboard_log(batch_visualize, model, val_set, sup_loader, ep, 100, 200, kwargs['n_heads'],
                            kwargs['logdir'], kwargs['w_size'], writer, True, kwargs['VAT_start'], kwargs['reconstruction'])

        if ep % 200 == 0:
            torch.save(model.state_dict(), os.path.join(kwargs['logdir'], f"model-{ep}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(kwargs['logdir'], "last-optimizer-state.pt"))

        for key, val in losses.items():
            writer.add_scalar(key, val.item(), global_step=ep)

    print("[INFO] Training complete. Evaluating...")
    with torch.no_grad():
        model.eval()
        metrics = evaluate_wo_velocity(tqdm(full_val), model, reconstruction=False,
                                       save_path=os.path.join(kwargs['logdir'], 'MIDI_results'))

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, cat, name = key.split('/')
            print(f'{cat:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')

    with open(os.path.join(kwargs['logdir'], 'result_dict'), 'wb') as f:
        pickle.dump(metrics, f)


def main():
    config = get_config()
    train(**config)

if __name__ == '__main__':
    main()
