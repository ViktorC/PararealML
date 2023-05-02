import numpy as np
import tensorflow as tf

from pararealml import DiscreteInitialCondition
from pararealml.operators.ml.physics_informed import DataArgs, OptimizationArgs
from pararealml.utils.rand import SEEDS, set_random_seed
from pararealml.utils.time import time

if __name__ == "__main__":
    set_random_seed(SEEDS[0])

    from experiments.burgers.ivp import ivp
    from experiments.burgers.operators import coarse_fdm, coarse_piml

    y_0s = np.load("data/piml_initial_conditions.npy")
    y_0_functions = [
        DiscreteInitialCondition(
            ivp.constrained_problem, y_0, coarse_fdm.vertex_oriented
        ).y_0
        for y_0 in y_0s
    ]
    training_y_0_functions = y_0_functions[:4000]
    validation_y_0_functions = y_0_functions[4000:5000]
    test_y_0_functions = y_0_functions[5000:]

    time("piml training")(coarse_piml.train)(
        ivp.constrained_problem,
        (0.0, coarse_piml.d_t),
        training_data_args=DataArgs(
            y_0_functions=training_y_0_functions,
            n_domain_points=16000,
            n_boundary_points=6000,
            n_batches=40000,
            n_ic_repeats=20,
        ),
        validation_data_args=DataArgs(
            y_0_functions=validation_y_0_functions,
            n_domain_points=16000,
            n_boundary_points=6000,
            n_batches=5000,
            n_ic_repeats=20,
            shuffle=False,
        ),
        test_data_args=DataArgs(
            y_0_functions=test_y_0_functions,
            n_domain_points=16000,
            n_boundary_points=6000,
            n_batches=5000,
            n_ic_repeats=20,
            shuffle=False,
        ),
        optimization_args=OptimizationArgs(
            optimizer=tf.optimizers.Adam(
                learning_rate=tf.optimizers.schedules.ExponentialDecay(
                    1e-3, decay_steps=800000, decay_rate=np.sqrt(1e-3)
                )
            ),
            epochs=40,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0,
                    patience=10,
                    restore_best_weights=True,
                )
            ],
        ),
    )
    coarse_piml.model.model.save_weights("weights/piml")
