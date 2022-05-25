from DistCurve.estimate import getModel
def DistCurve(x, x1):
    model = getModel()
    model_path =
    model.load_weights(args.model_path, by_name=True)
    model.compile(optimizer="Adam", loss="mean_absolute_error")
    model.summary(print_fn=lambda x: print(x, file=f))
    features = np.load(args.features_path)
    predictions = model.predict(features)
    if args.labels_path:
        labels = np.load(args.labels_path)
        mae = np.mean(np.abs(labels - predictions))
        print("MAE: {}\n".format(mae), file=f)
    print("Predictions:", file=f)
    for prediction in predictions:
        print(prediction[0], file=f)