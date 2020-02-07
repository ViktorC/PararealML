#include <math.h>
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

void simulate_rabbit_population() {
    const double n0 = 1000;
    const double duration = 10;
    const double d_t = .05;
    const double r = d_rabbit_population_wrt_t(0, 1);

    double n_g = n0;
    double n_f = n0;
    double t;
    for (t = 0; t <= duration; t += d_t) {
        n_g = euler(n_g, d_rabbit_population_wrt_t(t, n_g), d_t);
        n_f = runge_kutta_4(n_f, t, d_rabbit_population_wrt_t, d_t);
        double n_a = n0 * exp(r * (t + d_t));
        printf("coarse: %.3f; fine: %.3f; analytic: %.3f\n", n_g, n_f, n_a);
    }
}

int main(int argc, char** argv) {
    simulate_rabbit_population();
}
