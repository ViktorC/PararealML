#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>


double gaussian(double x, double mu, double sigma) {
    return pow(M_E, -(x - mu) * (x - mu) / (sigma * sigma)) /
            sigma * sqrt(2 * M_PI);
}

double second_order_finite_diff(double u, double u_prev, double u_next,
        double d_x) {
    return (u_prev - 2 * u + u_next) / (d_x * d_x);
}

double euler(double y, double d_y_wrt_t, double d_t) {
    return y + d_t * d_y_wrt_t;
}

double runge_kutta_4(double y, double t, double (*d_y_wrt_t)(double, double), double d_t) {
    double k1 = d_t * d_y_wrt_t(t, y);
    double k2 = d_t * d_y_wrt_t(t + d_t / 2, y + k1 / 2);
    double k3 = d_t * d_y_wrt_t(t + d_t / 2, y + k2 / 2);
    double k4 = d_t * d_y_wrt_t(t + d_t, y + k3);
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
}

void simulate_1d_diffusion() {
    const double a = -3;
    const double b = 3;
    const double mu = 0;
    const double sigma = 1;
    const double diff_coeff = .5;
    const double duration = 10;
    const double d_t = .05;
    const double d_x = .25;

    const int grid_size = (int) ((b - a) / d_x) + 1;
    double* u = malloc(grid_size * sizeof(double));
    double* d2_u = malloc(grid_size * sizeof(double));

    int i;
    for (i = 0; i < grid_size; i++) {
        if (i == 0) {
            u[i] = 0;
        } else if (i == grid_size - 1) {
            u[i] = 0;
        } else {
            u[i] = gaussian(a + i * d_x, mu, sigma);
        }
        printf("%.3f ", u[i]);
    }
    printf("\n");

    double t;
    for (t = 0; t <= duration; t += d_t) {
        for (i = 1; i < grid_size - 1; i++) {
            d2_u[i] = second_order_finite_diff(u[i], u[i - 1], u[i + 1], d_x);
        }
        double sum = 0;
        for (i = 0; i < grid_size; i++) {
            sum += u[i];
            if (i > 0 && i < grid_size - 1) {
                u[i] = euler(u[i], diff_coeff * d2_u[i], d_t);
            }
            printf("%.3f ", u[i]);
        }
        printf("\t\t%.3f\n", sum);
    }

    free(u);
    free(d2_u);
}

double d_rabbit_population_wrt_t(double t, double n) {
    static const double r = .01;
    return r * n;
}

typedef struct {
    int start_ind;
    int time_steps;
    double d_t;
    double* ns;
    double* corrections;
} parareal_thread_args;

void* correct_estimates(void* args) {
    parareal_thread_args* thread_args = (parareal_thread_args*) args;
    int i;
    for (i = thread_args->start_ind; i < thread_args->start_ind + thread_args->time_steps; i++) {
        double t = i * thread_args->d_t;
        double n = thread_args->ns[i];
        double f = runge_kutta_4(n, t, d_rabbit_population_wrt_t, thread_args->d_t);
        double g = euler(n, d_rabbit_population_wrt_t(t, n), thread_args->d_t);
        thread_args->corrections[i] = f - g;
    }
    return NULL;
}

void print_array(double* array, int size) {
    int i;
    for (i = 0; i < size; i++) {
        printf("%.3f ", array[i]);
    }
    printf("\n");
}

void simulate_rabbit_population() {
    const double n0 = 10000;
    const double r = d_rabbit_population_wrt_t(0, 1);

    const double duration = 20;
    const double d_t = .5;
    const int time_steps = (int) (duration / d_t);

    const int k = 2;
    const double n_threads = 4;
    const int time_steps_per_thread = time_steps / n_threads;

    double* ns = malloc(time_steps * sizeof(double));
    double* corrections = malloc(time_steps * sizeof(double));
    double* analytic_ns = malloc(time_steps * sizeof(double));

    ns[0] = n0;
    corrections[0] = n0;
    analytic_ns[0] = n0;

    int i;
    for (i = 1; i < time_steps; i++) {
        double t = i * d_t;
        analytic_ns[i] = n0 * exp(r * t);
    }

    for (i = 0; i < time_steps - 1; i++) {
        double t = i * d_t;
        ns[i + 1] = euler(ns[i], d_rabbit_population_wrt_t(t, ns[i]), d_t);
    }

    printf("Coarse solution:\n");
    print_array(ns, time_steps);

    pthread_t* threads = malloc(n_threads * sizeof(pthread_t));
    parareal_thread_args* thread_args = malloc(n_threads * sizeof(parareal_thread_args));

    for (i = 0; i < n_threads; i++) {
        thread_args[i].start_ind = i * time_steps_per_thread;
        thread_args[i].time_steps = time_steps_per_thread;
        thread_args[i].d_t = d_t;
        thread_args[i].ns = ns;
        thread_args[i].corrections = corrections;
    }

    for (i = 0; i < k; i++) {
        int rc;
        int j;
        for (j = 0; j < n_threads; j++) {
            rc = pthread_create(&threads[j], NULL, correct_estimates, &thread_args[j]);
            assert(!rc);
        }
        for (j = 0; j < n_threads; j++) {
            rc = pthread_join(threads[j], NULL);
            assert(!rc);
        }
        for (j = 0; j < time_steps - 1; j++) {
            double t = j * d_t;
            ns[j + 1] = euler(ns[j], d_rabbit_population_wrt_t(t, ns[j]), d_t) + corrections[j];
        }
        printf("Fine solution %d:\n", i + 1);
        print_array(ns, time_steps);
    }

    printf("Analytic solution:\n");
    print_array(analytic_ns, time_steps);

    free(threads);
    free(thread_args);

    free(ns);
    free(corrections);
    free(analytic_ns);
}

int main(int argc, char** argv) {
    simulate_rabbit_population();
}
