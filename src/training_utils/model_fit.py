




def fit_model(
        CFG,
        model,
        train_ds,
        valid_ds,
        callbacks,
        steps_per_epoch,
        num_valid,
        ):
    history = model.fit(
                    train_ds,
                    epochs=CFG.epoch-CFG.resume,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=callbacks,
                    validation_data=valid_ds,
                    verbose=CFG.verbose,
                    validation_steps=-(num_valid//-CFG.batch_size)
                )
    return history
    