from sklearn.metrics import mean_absolute_error

def evaluate(df_val, predictions):
    return mean_absolute_error(df_val.loss, predictions)