import numpy as np
import tensorflow as tf

from pararealml.operators.ml.physics_informed import DataArgs, OptimizationArgs
from pararealml.utils.rand import SEEDS, set_random_seed
from pararealml.utils.time import time

if __name__ == "__main__":
    set_random_seed(SEEDS[0])

    from experiments.van_der_pol.ivp import cp
    from experiments.van_der_pol.operators import coarse_piml

    y_0s = np.load("data/piml_initial_conditions.npy")
    y_0_functions = [lambda _, _y_0=y_0: _y_0 for y_0 in y_0s]
    training_y_0_functions = y_0_functions[:3200]
    validation_y_0_functions = y_0_functions[3200:4000]
    test_y_0_functions = y_0_functions[4000:]

    time("piml training")(coarse_piml.train)(
        cp,
        (0.0, coarse_piml.d_t),
        training_data_args=DataArgs(
            y_0_functions=training_y_0_functions,
            n_domain_points=4000,
            n_batches=400,
            n_ic_repeats=400,
        ),
        validation_data_args=DataArgs(
            y_0_functions=validation_y_0_functions,
            n_domain_points=4000,
            n_batches=100,
            n_ic_repeats=100,
            shuffle=False,
        ),
        test_data_args=DataArgs(
            y_0_functions=test_y_0_functions,
            n_domain_points=4000,
            n_batches=125,
            n_ic_repeats=125,
            shuffle=False,
        ),
        optimization_args=OptimizationArgs(
            optimizer=tf.optimizers.Adam(
                learning_rate=tf.optimizers.schedules.ExponentialDecay(
                    2.5e-3, decay_steps=60000, decay_rate=0.1
                )
            ),
            epochs=360,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0,
                    patience=90,
                    restore_best_weights=True,
                    verbose=1,
                )
            ],
        ),
    )
    coarse_piml.model.model.save_weights("weights/piml")
