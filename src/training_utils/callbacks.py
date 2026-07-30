



def build_callbacks(
    CFG,
    fold,
    train_ds,
    valid_ds,
    num_valid,
    strategy
    ):
    logger = tf.keras.callbacks.CSVLogger(f'{CFG.output_dir}/{CFG.comment}-fold{fold}-logs.csv')
    sv_loss = tf.keras.callbacks.ModelCheckpoint(f'{CFG.output_dir}/{CFG.comment}-fold{fold}-best.h5',monitor='val_loss',verbose =0,save_best_only=True,
            save_weights_only=True,mode = 'min',save_freq = 'epoch')
    snap = Snapshot(f'{CFG.output_dir}/{CFG.comment}-fold{fold}',CFG.snapshot_epochs)
    swa = SWA(f'{CFG.output_dir}/{CFG.comment}-fold{fold}',CFG.swa_epochs,strategy = strategy, train_ds = train_ds, valid_ds= valid_ds, valid_steps =-(num_valid//-CFG.batch_size))
    callbacks = []

    if CFG.save_output:
        callbacks.append(logger)
        callbacks.append(snap)
        callbacks.append(swa)
        if fold != 'all':
            callbacks.append(sv_loss)
    return callbacks