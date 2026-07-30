



def evaluate_model(
                CFG,
                model,
                fold,
                valid_ds,
                num_valid,
            ):
    if CFG.save_output:
        try:
            model.load_weights(f'{CFG.output_dir}/{CFG.comment}-fold{fold}-best.h5')
        except:
            pass
    if fold != 'all':
        cv = model.evaluate(valid_ds,verbose= CFG.verbose,steps =-(num_valid//-CFG.batch_size))
    else:
        cv = None

    return cv