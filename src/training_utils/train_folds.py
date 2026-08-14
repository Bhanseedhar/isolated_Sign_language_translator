

from .build_data_split import train_val_split
from .build_model import build_model
from .callbacks import build_callbacks
from .evaluate_model import evaluate_model
from .model_fit import fit_model
from .resume_training import resume_training
from .training_env_setup import setup_training_env
from .training_summary import show_training_summary
from src.config import CFG
from .schedules import WeightDecayScheduler


def train_fold(CFG,fold,train_files,valid_files,strategy,summary=True,):

    setup_training_env(CFG)

    train_ds, valid_ds, num_train, num_valid, steps_per_epoch,valid_files = train_val_split(CFG,fold,train_files,valid_files)
    model,schedule,decay_schedule = build_model(CFG,strategy,steps_per_epoch)

    if summary:
        show_training_summary(model,train_ds,valid_ds,schedule)

    if CFG.resume:
        resume_training(CFG, 
                        model, 
                        fold, 
                        train_ds, 
                        valid_ds, 
                        steps_per_epoch
                        )

    callbacks = build_callbacks(CFG,
                                fold,
                                train_ds,
                                valid_ds,
                                num_valid,
                                strategy
                                )

    #callbacks.append(WeightDecayScheduler(decay_schedule))
    
    history = fit_model(CFG,
                        model,
                        train_ds,
                        valid_ds,
                        callbacks,
                        steps_per_epoch,
                        num_valid,
                        )

    cv = evaluate_model(CFG,
                        model,
                        fold,
                        valid_ds,
                        num_valid,
                        )

    return model, cv, history

