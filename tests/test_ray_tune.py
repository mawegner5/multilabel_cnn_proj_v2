from ray import tune

def train(config):
    for i in range(10):
        # Simulating training logic
        loss = i * 0.1  # Example metric
        accuracy = i * 0.01  # Example metric
        tune.report(loss=loss, accuracy=accuracy)  # Report metrics

if __name__ == "__main__":
    analysis = tune.run(
        train,
        config={
            "parameter": 1  # Example hyperparameter
        },
        num_samples=1  # Run the training function once
    )

    print("Best config: ", analysis.best_config)
    print("Best trial final validation loss: ", analysis.best_result["loss"])
    print("Best trial final validation accuracy: ", analysis.best_result["accuracy"])
