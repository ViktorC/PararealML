import warnings
from multiprocessing import Process, Queue, connection
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.initial_condition import DiscreteInitialCondition
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.operator import Operator, discretize_time_domain
from pararealml.solution import Solution


class SupervisedMLOperator(Operator):
    """
    A supervised machine learning operator that models a high fidelity operator
    for solving initial value problems.
    """

    def __init__(
        self,
        d_t: float,
        vertex_oriented: bool,
        auto_regressive: bool = True,
        time_variant: bool = False,
        input_d_t: bool = False,
    ):
        """
        :param d_t: the temporal step size to use
        :param vertex_oriented: whether the operator is to evaluate the
            solutions of IVPs at the vertices or cell centers of the spatial
            meshes
        :param auto_regressive: whether to use the operator in auto-regressive
            mode
        :param time_variant: whether the time value should be used as a
            predictor
        :param input_d_t: whether to use the time step of the operator as an
            input to the model
        """
        if not auto_regressive and not time_variant:
            raise ValueError(
                "operator must be time variant if auto-regression is disabled"
            )

        if time_variant and input_d_t:
            raise ValueError(
                "operator must be time invariant to use d_t as an input"
            )

        super(SupervisedMLOperator, self).__init__(d_t, vertex_oriented)
        self._auto_regressive = auto_regressive
        self._time_variant = time_variant
        self._input_d_t = input_d_t
        self._model: Optional[Any] = None

    @property
    def auto_regressive(self) -> bool:
        """
        Whether the operator is auto-regressive.
        """
        return self._auto_regressive

    @property
    def time_variant(self) -> bool:
        """
        Whether the operator uses time as a predictor of the solution at the
        next time step.
        """
        return self._time_variant

    @property
    def input_d_t(self) -> bool:
        """
        Whether to use the time step of the operator as an input to the model.
        """
        return self._input_d_t

    @property
    def model(self) -> Optional[Any]:
        """
        The regression model behind the operator.
        """
        return self._model

    @model.setter
    def model(self, model: Optional[Any]):
        self._model = model

    def solve(
        self, ivp: InitialValueProblem, parallel_enabled: bool = True
    ) -> Solution:
        if self._model is None:
            raise ValueError("operator has no model")

        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation

        y_shape = cp.y_shape(self._vertex_oriented)

        inputs = self._create_input_placeholder(cp)
        t = discretize_time_domain(ivp.t_interval, self._d_t)[1:]
        y = np.empty((len(t),) + y_shape)

        y_0 = ivp.initial_condition.discrete_y_0(self._vertex_oriented)

        for i, t_i in enumerate(t):
            inputs[
                :,
                : inputs.shape[1]
                - diff_eq.x_dimension
                - (self._time_variant or self._input_d_t),
            ] = y_0.reshape((1, -1))
            if self._time_variant:
                inputs[:, -diff_eq.x_dimension - 1] = t_i
            elif self._input_d_t:
                inputs[:, -diff_eq.x_dimension - 1] = self._d_t

            y_i = self._model.predict(inputs)
            y[i, ...] = y_i.reshape(y_shape)

            if self._auto_regressive:
                y_0 = y_i

        return Solution(
            ivp, t, y, vertex_oriented=self._vertex_oriented, d_t=self._d_t
        )

    def generate_data(
        self,
        ivp: InitialValueProblem,
        oracle: Operator,
        iterations: int,
        perturbation_function: Callable[[float, np.ndarray], np.ndarray],
        isolate_perturbations: bool = False,
        repeat_on_error: bool = False,
        n_jobs: int = 1,
        seeds: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates data to train an operator model by using the oracle to
        repeatedly solve sub-IVPs with perturbed initial conditions and a time
        domain extent matching the step size of this operator.

        :param ivp: the IVP to train the regression model on
        :param oracle: the operator providing the training data
        :param iterations: the number of data generation iterations
        :param perturbation_function: a function that takes a time argument,
            representing the start of an IVP's time domain, and the discrete
            initial conditions for the IVP and returns a perturbed version of
            the initial conditions
        :param isolate_perturbations: whether to stop perturbations from
            propagating through to the subsequent sub-IVPs if in
            auto-regressive mode
        :param repeat_on_error: whether to repeat the initial condition
            perturbation and the target computation for any IVP if an error
            is raised
        :param n_jobs: the number of parallel processes to use for the data
            generation; if it is greater than one, all arguments of this method
            must be pickleable
        :param seeds: a sequence of NumPy random seeds to use in the data
            generation processes; the length of this sequence must match the
            number of jobs
        :return: a tuple of the inputs and the target outputs
        """
        if iterations <= 0:
            raise ValueError("number of iterations must be greater than 0")
        if n_jobs < 1:
            raise ValueError("number of jobs must be greater than 0")
        if seeds is not None:
            if len(seeds) != n_jobs:
                raise ValueError(
                    f"number of seeds ({len(seeds)}) must match "
                    f"number of jobs ({n_jobs})"
                )
        else:
            seeds = [None] * n_jobs

        queue: Queue[Tuple[int, np.ndarray, np.ndarray]] = Queue()

        if n_jobs == 1:
            self._generate_data(
                0,
                ivp,
                oracle,
                iterations,
                perturbation_function,
                isolate_perturbations,
                repeat_on_error,
                seeds[0],
                queue,
            )
            return queue.get()[1:]

        model = self._model
        self._model = None

        try:
            process_sentinels = []
            for process_rank, iterations_indices in enumerate(
                np.array_split(np.arange(iterations), n_jobs)
            ):
                process = Process(
                    target=self._generate_data,
                    args=(
                        process_rank,
                        ivp,
                        oracle,
                        len(iterations_indices),
                        perturbation_function,
                        isolate_perturbations,
                        repeat_on_error,
                        seeds[process_rank],
                        queue,
                    ),
                )
                process.daemon = True
                process.start()
                process_sentinels.append(process.sentinel)

            generated_data = [queue.get() for _ in range(n_jobs)]
            connection.wait(process_sentinels)

            all_inputs: List[Optional[np.ndarray]] = [None] * n_jobs
            all_targets: List[Optional[np.ndarray]] = [None] * n_jobs
            for (rank, inputs, targets) in generated_data:
                all_inputs[rank] = inputs
                all_targets[rank] = targets

            return np.concatenate(all_inputs, axis=0), np.concatenate(
                all_targets, axis=0
            )

        finally:
            self._model = model

    def fit_model(
        self,
        model: Any,
        data: Tuple[np.ndarray, np.ndarray],
        test_size: float = 0.2,
        score_func: Callable[
            [np.ndarray, np.ndarray], float
        ] = mean_squared_error,
    ) -> Tuple[float, Optional[float]]:
        """
        Fits the regression model to the training share of the provided data
        points using random splitting, it stores the fitted model as a member
        variable for solving IVPs, and it returns the loss of the model
        evaluated on both the training and test data sets.

        :param model: the regression model to train
        :param data: a tuple of the inputs and the target outputs
        :param test_size: the fraction of all data points that should be used
            for testing
        :param score_func: the prediction scoring function to use
        :return: the training and test scores
        """
        if test_size:
            x_train, x_test, y_train, y_test = train_test_split(
                data[0],
                data[1],
                test_size=test_size,
            )
        else:
            shuffled_indices = np.random.permutation(len(data[0]))
            x_train = data[0][shuffled_indices]
            y_train = data[1][shuffled_indices]
            x_test = y_test = None

        model.fit(x_train, y_train)
        self._model = model

        y_train_hat = model.predict(x_train)
        train_score = score_func(y_train, y_train_hat)

        if test_size:
            y_test_hat = model.predict(x_test)
            test_score = score_func(y_test, y_test_hat)
        else:
            test_score = None

        return train_score, test_score

    def train(
        self,
        ivp: InitialValueProblem,
        oracle: Operator,
        model: Any,
        iterations: int,
        perturbation_function: Callable[[float, np.ndarray], np.ndarray],
        isolate_perturbations: bool = False,
        repeat_on_error: bool = False,
        n_jobs: int = 1,
        seeds: Optional[Sequence[int]] = None,
        test_size: float = 0.2,
        score_func: Callable[
            [np.ndarray, np.ndarray], float
        ] = mean_squared_error,
    ) -> Tuple[float, Optional[float]]:
        """
        Fits a regression model to training data generated by the oracle.

        The inputs of the model are time t, spatial coordinates x, and the
        values of the solution at all points of the IVP's mesh at time t - d_t
        (for ODEs, no spatial coordinates are included in the features and the
        solution is evaluated merely at t). The model outputs are the predicted
        values of the solution at t and x (for ODEs, it is again just the
        solution at t). The time t is an optional input, for problems with no
        explicit dependence on t in the right hand sides of the differential
        equation or the boundary conditions, it may be omitted. In this case,
        the solution is modelled as a function of only the initial conditions
        (and x for PDEs).

        The training data is generated by using the oracle to repeatedly solve
        sub-IVPs with perturbed initial conditions and a time domain extent
        matching the step size of this operator.

        :param ivp: the IVP to train the regression model on
        :param oracle: the operator providing the training data
        :param model: the model to fit to the training data
        :param iterations: the number of data generation iterations
        :param perturbation_function: a function that takes a time argument,
            representing the start of an IVP's time domain, and the discrete
            initial conditions for the IVP and returns a perturbed version of
            the initial conditions
        :param isolate_perturbations: whether to stop perturbations from
            propagating through to the subsequent sub-IVPs if in
            auto-regressive mode
        :param repeat_on_error: whether to repeat the initial condition
            perturbation and the target computation for any IVP if an error
            is raised
        :param n_jobs: the number of parallel processes to use for the data
            generation; if it is greater than one, all arguments relating to
            the data generation must be pickleable
        :param seeds: a sequence of NumPy random seeds to use in the data
            generation processes; the length of this sequence must match the
            number of jobs
        :param test_size: the fraction of all data points that should be used
            for testing
        :param score_func: the prediction scoring function to use
        :return: the training and test scores
        """
        data = self.generate_data(
            ivp,
            oracle,
            iterations,
            perturbation_function,
            isolate_perturbations=isolate_perturbations,
            repeat_on_error=repeat_on_error,
            n_jobs=n_jobs,
            seeds=seeds,
        )
        return self.fit_model(
            model, data, test_size=test_size, score_func=score_func
        )

    def _create_input_placeholder(self, cp: ConstrainedProblem) -> np.ndarray:
        """
        Creates a placeholder array for the model inputs. If the constrained
        problem is a PDE, it pre-populates the first x_dimension columns
        corresponding to the spatial coordinates.

        :param cp: the constrained problem to base the inputs on
        :return: the placeholder array for the model inputs
        """
        diff_eq = cp.differential_equation
        if not diff_eq.x_dimension:
            return np.empty((1, diff_eq.y_dimension + self._time_variant))

        x = cp.mesh.all_index_coordinates(self._vertex_oriented, flatten=True)
        y = np.empty((len(x), diff_eq.y_dimension * len(x)))

        if self._time_variant or self._input_d_t:
            t = np.empty((len(x), 1))
            return np.hstack([y, t, x])
        else:
            return np.hstack([y, x])

    def _generate_data(
        self,
        rank: int,
        ivp: InitialValueProblem,
        oracle: Operator,
        iterations: int,
        perturbation_function: Callable[[float, np.ndarray], np.ndarray],
        isolate_perturbations: bool,
        repeat_on_error: bool,
        seed: Optional[int],
        queue: Queue,
    ):
        """
        Generates data to train an operator model sequentially by using the
        oracle to repeatedly solve sub-IVPs with perturbed initial conditions
        and a time domain extent matching the step size of this operator.

        :param rank: the rank of the executing process
        :param ivp: the IVP to train the regression model on
        :param oracle: the operator providing the training data
        :param iterations: the number of data generation iterations
        :param perturbation_function: a function that takes a time argument,
            representing the start of an IVP's time domain, and the discrete
            initial conditions for the IVP and returns a perturbed version of
            the initial conditions
        :param isolate_perturbations: whether to stop perturbations from
            propagating through to the subsequent sub-IVPs if in
            auto-regressive mode
        :param repeat_on_error: whether to repeat the initial condition
            perturbation and the target computation for any IVP if an error
            is raised
        :param seed: the NumPy random seed to use for the data generation
        :param queue: a queue to add the results to in the form of a tuple of
            the process rank, the inputs, and the target outputs
        """
        if seed is not None:
            np.random.seed(seed)

        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation
        x_dim = diff_eq.x_dimension
        y_dim = diff_eq.y_dimension

        t = discretize_time_domain(ivp.t_interval, self._d_t)
        y_0 = ivp.initial_condition.discrete_y_0(self._vertex_oriented)

        unperturbed_sub_y_0s: Optional[np.ndarray] = None
        if self._auto_regressive and isolate_perturbations:
            unperturbed_sub_y_0s = self._perturb_and_solve_ivp(
                InitialValueProblem(
                    ivp.constrained_problem,
                    (t[0], t[-2]),
                    ivp.initial_condition,
                ),
                lambda _, y: y,
                oracle,
                False,
            ).discrete_y(self._vertex_oriented)[
                np.rint((t[1:-1] - t[0]) / oracle.d_t).astype(int) - 1,
                ...,
            ]

        single_time_point_inputs = self._create_input_placeholder(cp)
        n_spatial_points = single_time_point_inputs.shape[0]
        single_epoch_inputs = np.tile(
            single_time_point_inputs, (len(t) - 1, 1)
        )
        if self._time_variant:
            single_epoch_inputs[:, -x_dim - 1] = np.repeat(
                t[1:], n_spatial_points
            )
        elif self._input_d_t:
            single_epoch_inputs[:, -x_dim - 1] = self._d_t

        inputs = np.tile(single_epoch_inputs, (iterations, 1))
        targets = np.empty((inputs.shape[0], y_dim))
        for iteration in range(iterations):
            offset = iteration * n_spatial_points * (len(t) - 1)

            if self._auto_regressive:
                y_i = y_0
                for i, t_i in enumerate(t[:-1]):
                    perturbed_sub_ivp_solution = self._perturb_and_solve_ivp(
                        InitialValueProblem(
                            cp,
                            (t_i, t_i + self._d_t),
                            DiscreteInitialCondition(
                                cp, y_i, self._vertex_oriented
                            ),
                        ),
                        perturbation_function,
                        oracle,
                        repeat_on_error,
                    )
                    perturbed_sub_ivp = (
                        perturbed_sub_ivp_solution.initial_value_problem
                    )
                    perturbed_y_i = (
                        perturbed_sub_ivp.initial_condition.discrete_y_0(
                            self._vertex_oriented
                        )
                    )
                    perturbed_y_next = perturbed_sub_ivp_solution.discrete_y(
                        self._vertex_oriented
                    )[-1]
                    t_offset = offset + i * n_spatial_points
                    inputs[
                        t_offset : t_offset + n_spatial_points,
                        : y_dim * n_spatial_points,
                    ] = perturbed_y_i.reshape((1, -1))
                    targets[
                        t_offset : t_offset + n_spatial_points, :
                    ] = perturbed_y_next.reshape((-1, y_dim))
                    y_i = (
                        unperturbed_sub_y_0s[i]
                        if isolate_perturbations and i < len(t) - 2
                        else perturbed_y_next
                    )

            else:
                perturbed_ivp_solution = self._perturb_and_solve_ivp(
                    ivp,
                    perturbation_function,
                    oracle,
                    repeat_on_error,
                )
                perturbed_ivp = perturbed_ivp_solution.initial_value_problem
                perturbed_y_0 = perturbed_ivp.initial_condition.discrete_y_0(
                    self._vertex_oriented
                )
                perturbed_y = perturbed_ivp_solution.discrete_y(
                    self._vertex_oriented
                )
                inputs[
                    offset : offset + (len(t) - 1) * n_spatial_points,
                    : inputs.shape[1] - x_dim - self._time_variant,
                ] = perturbed_y_0.reshape((1, -1))
                targets[
                    offset : offset + (len(t) - 1) * n_spatial_points, :
                ] = perturbed_y[
                    np.rint((t[1:] - t[0]) / oracle.d_t).astype(int) - 1, ...
                ].reshape(
                    (-1, y_dim)
                )

        queue.put((rank, inputs, targets))

    def _perturb_and_solve_ivp(
        self,
        ivp: InitialValueProblem,
        perturbation_function: Callable[[float, np.ndarray], np.ndarray],
        oracle: Operator,
        repeat_on_error: bool,
    ) -> Solution:
        """
        Takes the provided IVP, perturbs its initial conditions, and solves the
        perturbed IVP.

        :param ivp: the IVP to solve after perturbing its initial conditions
        :param perturbation_function: a function that takes a time argument,
            representing the start of the IVP's time domain, and the discrete
            initial conditions for the IVP and returns a perturbed version
            of the initial conditions
        :param oracle: the operator to solve the perturbed IVP with
        :param repeat_on_error: whether to repeat the initial condition
            perturbation and the IVP solution if an exception is raised
        :return: the solution of the perturbed IVP
        """
        while True:
            y_0 = ivp.initial_condition.discrete_y_0(self._vertex_oriented)
            perturbed_y_0 = perturbation_function(ivp.t_interval[0], y_0)
            if perturbed_y_0.shape != y_0.shape:
                raise ValueError(
                    f"perturbed y shape {perturbed_y_0.shape} must "
                    f"match input y shape {y_0.shape}"
                )

            perturbed_ivp = InitialValueProblem(
                ivp.constrained_problem,
                ivp.t_interval,
                DiscreteInitialCondition(
                    ivp.constrained_problem,
                    perturbed_y_0,
                    self._vertex_oriented,
                ),
            )

            try:
                return oracle.solve(perturbed_ivp)
            except Exception as exception:
                if repeat_on_error:
                    warnings.warn(
                        "Failed to solve IVP with perturbed initial "
                        f"conditions; {str(exception)}"
                    )
                    continue

                raise exception
