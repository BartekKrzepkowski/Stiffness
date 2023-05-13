#!/usr/bin/env python3
import numpy as np
import torch

# from rich.traceback import install
# install(show_locals=True)

from src.utils.prepare import prepare_model, prepare_loaders, prepare_criterion, prepare_optim_and_scheduler
from src.utils.utils_trainer import manual_seed
from src.utils.utils_visualisation import ee_tensorboard_layout
from src.trainer.trainer_classification import TrainerClassification
from src.trainer.trainer_context import TrainerContext


def objective(batch_size, lr, width, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # model
    NUM_FEATURES = 3
    NUM_CLASSES = 10
    N = 1
    DIMS = [NUM_FEATURES] + [width] * (N + 1) + [128, NUM_CLASSES]
    CONV_PARAMS = {'img_height': 32, 'img_widht': 32, 'kernels': [3, 3] * (N + 1), 'strides': [1, 1] * (N + 1), 'paddings': [1, 1] * (N + 1), 'whether_pooling': [False, True] * (N + 1)}
    # trainer & schedule
    RANDOM_SEED = 83
    EPOCHS = 150
    GRAD_ACCUM_STEPS = 1
    CLIP_VALUE = 100.0

    # prepare params
    type_names = {
        'model': model_name,
        'criterion': 'fp',
        'dataset': 'cifar10',
        'optim': 'sgd',
        'scheduler': None
    }
    # wandb params
    GROUP_NAME = f'Stiffness_{batch_size}, lr_{lr}_dataset_{type_names["dataset"]}'
    EXP_NAME = f'{GROUP_NAME}_model_{type_names["model"]}, width_{width}_depth_{len(DIMS) - 3}'
    PROJECT_NAME = 'Stiffness' 
    ENTITY_NAME = 'ideas_cv'

    h_params_overall = {
        'model': {'layers_dim': DIMS, 'activation_name': 'relu', 'conv_params': CONV_PARAMS},
        'criterion': {'model': None, 'general_criterion_name': 'ce', 'num_classes': NUM_CLASSES,
                      'whether_record_trace': False, 'fpw': 0.0},
        'dataset': {'dataset_path': None, 'whether_aug': False},
        'loaders': {'batch_size': batch_size, 'pin_memory': True, 'num_workers': 8},
        'optim': {'lr': lr, 'momentum': 0.0, 'weight_decay': 0.0},
        'scheduler': {'eta_min': 1e-6, 'T_max': None},
        'type_names': type_names
    }
    # set seed to reproduce the results in the future
    manual_seed(random_seed=RANDOM_SEED, device=device)
    # prepare model
    model = prepare_model(type_names['model'], model_params=h_params_overall['model']).to(device)
    # prepare criterion
    h_params_overall['criterion']['model'] = model
    criterion = prepare_criterion(type_names['criterion'], h_params_overall['criterion'])
    # prepare loaders
    loaders = prepare_loaders(type_names['dataset'], h_params_overall['dataset'], h_params_overall['loaders'])
    # prepare optimizer & scheduler
    T_max = (len(loaders['train']) // GRAD_ACCUM_STEPS) * EPOCHS
    h_params_overall['scheduler']['T_max'] = T_max
    optim, lr_scheduler = prepare_optim_and_scheduler(model, type_names['optim'], h_params_overall['optim'],
                                                      type_names['scheduler'], h_params_overall['scheduler'])
    
    # DODAJ - POPRAWNE DANE
    print(sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()))
    x_data = torch.load(f'data/{type_names["dataset"]}_held_out_x.pt').to(device)
    y_data = torch.load(f'data/{type_names["dataset"]}_held_out_y.pt').to(device)   

    # prepare trainer
    params_trainer = {
        'model': model,
        'criterion': criterion,
        'loaders': loaders,
        'optim': optim,
        'lr_scheduler': lr_scheduler,
        'extra': {'x_true1': x_data, 'y_true1': y_data, 'num_classes': NUM_CLASSES},
    }
    trainer = TrainerClassification(**params_trainer)

    # prepare run
    params_names = [n for n, p in model.named_parameters() if p.requires_grad]
    config = TrainerContext(
        epoch_start_at=0,
        epoch_end_at=EPOCHS,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        save_multi=0,#T_max // 10,
        log_multi=1,#(T_max // EPOCHS) // 10,
        stiff_multi=(T_max // EPOCHS) // 2,
        clip_value=CLIP_VALUE,
        base_path='reports',
        exp_name=EXP_NAME,
        logger_config={'logger_name': 'tensorboard', 'project_name': PROJECT_NAME, 'entity': ENTITY_NAME,
                       'hyperparameters': h_params_overall, 'whether_use_wandb': True,
                       'layout': ee_tensorboard_layout(params_names), 'mode': 'online'
                       },
        whether_disable_tqdm=True,
        random_seed=RANDOM_SEED,
        extra={},
        device=device
    )
    trainer.run_exp(config)


if __name__ == "__main__":
    for lr in [1e-1]:
        for width in [96, 256]:
            for model_name in ['simple_cnn']:#, 'simple_cnn_with_bnorm_and_dropout']:
                objective(200, lr, width, model_name)