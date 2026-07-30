def show_training_summary(model,train_ds,valid_ds,schedule):
    print()
    model.summary()
    print()
    print(train_ds,valid_ds)
    print()
    schedule.plot()
    print()
    