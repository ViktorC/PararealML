import numpy as np
import tensorflow as tf

from pararealml.operators.ml.physics_informed import DataArgs, OptimizationArgs
from pararealml.utils.rand import SEEDS, set_random_seed
from pararealml.utils.time import time

if __name__ == "__main__":
    set_random_seed(SEEDS[0])

    from experiments.population_growth.ivp import cp
    from experiments.population_growth.operators import coarse_piml

    y_0s = np.load("data/piml_initial_conditions.npy")
    y_0_functions = [lambda _, _y_0=y_0: _y_0 for y_0 in y_0s]
    training_y_0_functions = y_0_functions[:1000]
    validation_y_0_functions = y_0_functions[1000:2000]
    test_y_0_functions = y_0_functions[2000:]

    time("piml training")(coarse_piml.train)(
        cp,
        (0.0, coarse_piml.d_t),
        training_data_args=DataArgs(
            y_0_functions=training_y_0_functions,
            n_domain_points=500,
            n_batches=4,
            n_ic_repeats=4,
            shuffle=False,
        ),
        validation_data_args=DataArgs(
            y_0_functions=validation_y_0_functions,
            n_domain_points=500,
            n_batches=4,
            n_ic_repeats=4,
            shuffle=False,
        ),
        test_data_args=DataArgs(
            y_0_functions=test_y_0_functions,
            n_domain_points=500,
            n_batches=4,
            n_ic_repeats=4,
            shuffle=False,
        ),
        optimization_args=OptimizationArgs(
            optimizer=tf.optimizers.Adam(
                learning_rate=tf.optimizers.schedules.ExponentialDecay(
                    5e-3, decay_steps=20000, decay_rate=0.1
                )
            ),
            epochs=10000,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0,
                    patience=2500,
                    restore_best_weights=True,
                    verbose=1,
                )
            ],
        ),
    )
    coarse_piml.model.model.save_weights("weights/piml")
