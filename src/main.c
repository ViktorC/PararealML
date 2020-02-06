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

double euler(double u, double d_u_wrt_t, double d_t) {
    return u + d_t * d_u_wrt_t;
}

int main(int argc, char** argv) {
    const double a = -3;
    const double b = 3;
    const double mu = 0;
    const double sigma = 1;
    const double diff_coeff = .5;
    const double duration = 10;
    const double d_t = .05;
    const double d_x = .25;

    const int grid_size = (int) ((b - a) / d_x);
    double* u = malloc(grid_size * sizeof(double));
    double* d2_u = malloc(grid_size * sizeof(double));

    int i;
    for (i = 0; i < grid_size; i++) {
        u[i] = gaussian(a + i * d_x, mu, sigma);
        printf("%.3f ", u[i]);
    }
    u[0] = 0.;
    u[grid_size - 1] = 0.;
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
