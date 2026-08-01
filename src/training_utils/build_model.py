from src.architecture_blocks.model_building import get_model
from src.training_utils.schedules import OneCycleLR
from src.training_utils.learners import FGM, AWP
import tensorflow as tf

def build_model(CFG,strategy,steps_per_epoch):
    with strategy.scope():
        dropout_step = CFG.dropout_start_epoch * steps_per_epoch


        model = get_model(
            max_len = CFG.max_len,
            dropout_step = dropout_step,
            dim = CFG.dim
        )


        schedule = OneCycleLR(
            CFG.lr,
            CFG.epoch, 
            warmup_epochs=CFG.epoch*CFG.warmup, 
            steps_per_epoch=steps_per_epoch, 
            resume_epoch=CFG.resume, 
            decay_epochs=CFG.epoch, 
            lr_min=CFG.lr_min, 
            decay_type=CFG.decay_type, 
            warmup_type='linear'
        )




        decay_schedule = OneCycleLR(
            CFG.lr*CFG.weight_decay, 
            CFG.epoch, 
            warmup_epochs=CFG.epoch*CFG.warmup, 
            steps_per_epoch=steps_per_epoch, 
            resume_epoch=CFG.resume, 
            decay_epochs=CFG.epoch, 
            lr_min=CFG.lr_min*CFG.weight_decay, 
            decay_type=CFG.decay_type, 
            warmup_type='linear'
        )


        awp_step = CFG.awp_start_epoch * steps_per_epoch
        
        
             
        
        
        if CFG.fgm:
            model = FGM(
                model.input,
                model.output,
                delta=CFG.awp_lambda,
                eps=0.,
                start_step=awp_step
            )
        elif CFG.awp:
            model = AWP(
                    model.input,
                    model.output,
                    delta=CFG.awp_lambda,
                    eps=0.,
                    start_step=awp_step,
                    late_dropout=model.late_dropout,
                    dropout_step=dropout_step,
                     )

        opt = tf.keras.optimizers.AdamW(
                learning_rate=schedule,
                weight_decay=decay_schedule,
                )

        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.CategoricalCrossentropy(
                from_logits=True,
                label_smoothing=0.1,
            ),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
            steps_per_execution=steps_per_epoch,
        )

    return model, schedule, decay_schedule