def resume_training(
            CFG, 
            model, 
            fold, 
            train_ds, 
            valid_ds, 
            steps_per_epoch
            ):

    
    print(f"----------resume from epoch {CFG.resume}-------------")
    model.load_weights(f'{CFG.output_dir}/{CFG.comment}-fold{fold}-last.h5')
    if train_ds is not None:
        model.evaluate(train_ds.take(steps_per_epoch))

    if valid_ds is not None:
        model.evaluate(valid_ds)

    
