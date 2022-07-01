from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from pararealml.constraint import Constraint, apply_constraints_along_last_axis
from pararealml.mesh import CoordinateSystem, Mesh

Slicer = List[Union[int, slice]]

BoundaryConstraintPair = Tuple[Optional[Constraint], Optional[Constraint]]


class NumericalDifferentiator(ABC):
    """
    A base class for numerical differentiators.
    """

    def __init__(self, tol: float = 1e-3):
        """
        :param tol: the stopping criterion for the Jacobi algorithm when
            computing the anti-Laplacian; once the second norm of the
            difference of the estimate and the updated estimate drops below
            this threshold, the equation is considered to be solved
        """
        if tol < 0.0:
            raise ValueError("tolerance must be non-negative")

        self._tol = tol

    @abstractmethod
    def _derivative(
        self,
        y: np.ndarray,
        d_x: float,
        x_axis: int,
        derivative_boundary_constraints: Union[
            Sequence[Optional[BoundaryConstraintPair]], np.ndarray
        ],
    ) -> np.ndarray:
        """
        Computes the derivative of y with respect to the spatial dimension
        defined by x_axis at every point of the mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step size of the mesh along the specified axis
        :param x_axis: the spatial dimension that y is to be differentiated
            with respect to
        :param derivative_boundary_constraints: a sequence of boundary
            constraint pairs with an element for each component of y that
            allows for applying constraints to the calculated first
            derivatives at the boundaries normal to the axis that y is
            differentiated with respect to
        :return: the derivative of y with respect to the spatial dimension
            defined by x_axis
        """

    @abstractmethod
    def _second_derivative(
        self,
        y: np.ndarray,
        d_x1: float,
        d_x2: float,
        x_axis1: int,
        x_axis2: int,
        derivative_boundary_constraints: Union[
            Sequence[Optional[BoundaryConstraintPair]], np.ndarray
        ],
    ) -> np.ndarray:
        """
        Computes the second derivative of y with respect to the spatial
        dimensions defined by x_axis1 and x_axis2 at every point of the mesh.

        :param y: the values of y at every point of the mesh
        :param d_x1: the step size of the mesh along the axis defined by
            x_axis1
        :param d_x2: the step size of the mesh along the axis defined by
            x_axis2
        :param x_axis1: the first spatial dimension that y is to be
            differentiated with respect to
        :param x_axis2: the second spatial dimension that y is to be
            differentiated with respect to
        :param derivative_boundary_constraints: a sequence of boundary
            constraint pairs with an element for each component of y
            that allows for applying constraints to the calculated first
            derivatives at the boundaries normal to the first axis of
            differentiation before computing the second derivatives
        :return: the second derivative of y with respect to the spatial
            dimensions defined by x_axis1 and x_axis2
        """

    @abstractmethod
    def _next_anti_laplacian_estimate(
        self,
        y_hat: np.ndarray,
        laplacian: np.ndarray,
        mesh: Mesh,
        derivative_boundary_constraints: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Given an estimate of the anti-Laplacian, it computes an improved
        estimate.

        :param y_hat: the current estimated values of the solution at every
            point of the mesh
        :param laplacian: the Laplacian for which y is to be determined
        :param mesh: the mesh representing the discretized spatial domain
        :param derivative_boundary_constraints: an optional 2D array
            (x dimension, y dimension) of boundary constraint pairs that
            specify constraints on the first derivatives of the solution
        :return: an improved estimate of y_hat
        """

    def gradient(
        self,
        y: np.ndarray,
        mesh: Mesh,
        x_axis: int,
        derivative_boundary_constraints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes the column of the Jacobian matrix of y corresponding to the
        spatial dimension specified by x_axis at every point of the mesh.

        :param y: the values of y at every point of the mesh
        :param mesh: the mesh representing the discretized spatial domain
        :param x_axis: the spatial dimension that y is to be differentiated
            with respect to
        :param derivative_boundary_constraints: a 2D array (x dimension,
            y dimension) of boundary constraint pairs that allow for applying
            constraints to the calculated first derivatives
        :return: the element(s) of the gradient of y corresponding to the
            specified axis
        """
        self._verify_input_shape_matches_mesh(y, mesh)
        if not (0 <= x_axis < mesh.dimensions):
            raise ValueError(
                f"x-axis ({x_axis}) must be non-negative and less than number "
                f"of x dimensions ({mesh.dimensions})"
            )

        derivative_boundary_constraints = (
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints, mesh.dimensions, y.shape[-1]
            )
        )

        derivative = self._derivative(
            y,
            mesh.d_x[x_axis],
            x_axis,
            derivative_boundary_constraints[x_axis],
        )

        if mesh.coordinate_system_type == CoordinateSystem.CARTESIAN:
            return derivative

        elif mesh.coordinate_system_type == CoordinateSystem.SPHERICAL:
            r = mesh.vertex_coordinate_grids[0][..., np.newaxis]
            phi = mesh.vertex_coordinate_grids[2][..., np.newaxis]
            if x_axis == 0:
                return derivative
            elif x_axis == 1:
                return derivative / (r * np.sin(phi))
            else:
                return derivative / r

        else:
            if x_axis == 1:
                r = mesh.vertex_coordinate_grids[0][..., np.newaxis]
                return derivative / r
            else:
                return derivative

    def hessian(
        self,
        y: np.ndarray,
        mesh: Mesh,
        x_axis1: int,
        x_axis2: int,
        derivative_boundary_constraints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes the column of the Hessian tensor of y corresponding to the
        spatial dimensions defined by x_axis1 and x_axis2 at every point of
        the mesh.

        :param y: the values of y at every point of the mesh
        :param mesh: the mesh representing the discretized spatial domain
        :param x_axis1: the first spatial dimension that y is to be
            differentiated with respect to
        :param x_axis2: the second spatial dimension that y is to be
            differentiated with respect to
        :param derivative_boundary_constraints: a 2D array (x dimension,
            y dimension) of boundary constraint pairs that allow for applying
            constraints to the calculated first derivatives before using them
            to compute the second derivatives
        :return: the element(s) of the Hessian of y corresponding to the
            specified axes
        """
        self._verify_input_shape_matches_mesh(y, mesh)
        if not (0 <= x_axis1 < mesh.dimensions) or not (
            0 <= x_axis2 < mesh.dimensions
        ):
            raise ValueError(
                f"both first x-axis ({x_axis1}) and second x-axis ({x_axis2}) "
                "must be non-negative and less than number of x dimensions "
                f"({mesh.dimensions})"
            )

        derivative_boundary_constraints = (
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints, mesh.dimensions, y.shape[-1]
            )
        )

        second_derivative = self._second_derivative(
            y,
            mesh.d_x[x_axis1],
            mesh.d_x[x_axis2],
            x_axis1,
            x_axis2,
            derivative_boundary_constraints[x_axis1],
        )

        if mesh.coordinate_system_type == CoordinateSystem.CARTESIAN:
            return second_derivative

        elif mesh.coordinate_system_type == CoordinateSystem.SPHERICAL:
            r = mesh.vertex_coordinate_grids[0][..., np.newaxis]
            phi = mesh.vertex_coordinate_grids[2][..., np.newaxis]

            if x_axis1 == 0 and x_axis2 == 0:
                return second_derivative

            elif x_axis1 == 1 and x_axis2 == 1:
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)
                d_y_over_d_r = self._derivative(
                    y, mesh.d_x[0], 0, derivative_boundary_constraints[0]
                )
                d_y_over_d_phi = self._derivative(
                    y, mesh.d_x[2], 2, derivative_boundary_constraints[2]
                )
                return (
                    d_y_over_d_r
                    + (second_derivative / sin_phi + cos_phi * d_y_over_d_phi)
                    / (r * sin_phi)
                ) / r

            elif x_axis1 == 2 and x_axis2 == 2:
                d_y_over_d_r = self._derivative(
                    y, mesh.d_x[0], 0, derivative_boundary_constraints[0]
                )
                return (second_derivative / r + d_y_over_d_r) / r

            elif (x_axis1 == 0 and x_axis2 == 1) or (
                x_axis1 == 1 and x_axis2 == 0
            ):
                d_y_over_d_theta = self._derivative(
                    y, mesh.d_x[1], 1, derivative_boundary_constraints[1]
                )
                return (second_derivative - d_y_over_d_theta / r) / (
                    r * np.sin(phi)
                )

            elif (x_axis1 == 0 and x_axis2 == 2) or (
                x_axis1 == 2 and x_axis2 == 0
            ):
                d_y_over_d_phi = self._derivative(
                    y, mesh.d_x[2], 2, derivative_boundary_constraints[2]
                )
                return (second_derivative - d_y_over_d_phi / r) / r

            else:
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)
                d_y_over_d_theta = self._derivative(
                    y, mesh.d_x[1], 1, derivative_boundary_constraints[1]
                )
                return (
                    sin_phi * second_derivative - cos_phi * d_y_over_d_theta
                ) / (r * sin_phi) ** 2

        else:
            r = mesh.vertex_coordinate_grids[0][..., np.newaxis]

            if (x_axis1 == 0 or x_axis1 == 2) and (
                x_axis2 == 0 or x_axis2 == 2
            ):
                return second_derivative

            elif x_axis1 == 1 and x_axis2 == 1:
                d_y_over_d_r = self._derivative(
                    y, mesh.d_x[0], 0, derivative_boundary_constraints[0]
                )
                return (second_derivative / r + d_y_over_d_r) / r

            elif (x_axis1 == 1 and x_axis2 == 0) or (
                x_axis1 == 0 and x_axis2 == 1
            ):
                d_y_over_d_theta = self._derivative(
                    y, mesh.d_x[1], 1, derivative_boundary_constraints[1]
                )
                return (second_derivative - d_y_over_d_theta / r) / r

            else:
                return second_derivative / r

    def divergence(
        self,
        y: np.ndarray,
        mesh: Mesh,
        derivative_boundary_constraints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes the divergence of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param mesh: the mesh representing the discretized spatial domain
        :param derivative_boundary_constraints: a 2D array (x dimension,
            y dimension) of boundary constraint pairs that allow for applying
            constraints to the calculated first derivatives before using them
            to compute the divergence
        :return: the divergence of y
        """
        self._verify_input_is_a_vector_field(y, mesh)

        derivative_boundary_constraints = (
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints, mesh.dimensions, y.shape[-1]
            )
        )

        if mesh.coordinate_system_type == CoordinateSystem.CARTESIAN:
            div = np.zeros(y.shape[:-1] + (1,))
            for i in range(y.shape[-1]):
                div += self._derivative(
                    y[..., i : i + 1],
                    mesh.d_x[i],
                    i,
                    derivative_boundary_constraints[i, i : i + 1],
                )

            return div

        elif mesh.coordinate_system_type == CoordinateSystem.SPHERICAL:
            r = mesh.vertex_coordinate_grids[0][..., np.newaxis]
            phi = mesh.vertex_coordinate_grids[2][..., np.newaxis]
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            y_r = y[..., :1]
            y_theta = y[..., 1:2]
            y_phi = y[..., 2:]
            d_y_r_over_d_r = self._derivative(
                y_r, mesh.d_x[0], 0, derivative_boundary_constraints[0, :1]
            )
            d_y_theta_over_d_theta = self._derivative(
                y_theta,
                mesh.d_x[1],
                1,
                derivative_boundary_constraints[1, 1:2],
            )
            d_y_phi_over_d_phi = self._derivative(
                y_phi, mesh.d_x[2], 2, derivative_boundary_constraints[2, 2:]
            )
            return (
                d_y_r_over_d_r
                + (
                    d_y_phi_over_d_phi
                    + 2.0 * y_r
                    + (d_y_theta_over_d_theta + cos_phi * y_phi) / sin_phi
                )
                / r
            )

        else:
            r = mesh.vertex_coordinate_grids[0][..., np.newaxis]
            y_r = y[..., :1]
            y_theta = y[..., 1:2]
            d_y_r_over_d_r = self._derivative(
                y_r, mesh.d_x[0], 0, derivative_boundary_constraints[0, :1]
            )
            d_y_theta_over_d_theta = self._derivative(
                y_theta,
                mesh.d_x[1],
                1,
                derivative_boundary_constraints[1, 1:2],
            )
            div = d_y_r_over_d_r + (y_r + d_y_theta_over_d_theta) / r

            if mesh.coordinate_system_type == CoordinateSystem.POLAR:
                return div
            else:
                y_z = y[..., 2:]
                d_y_z_over_d_z = self._derivative(
                    y_z, mesh.d_x[2], 2, derivative_boundary_constraints[2, 2:]
                )
                return div + d_y_z_over_d_z

    def curl(
        self,
        y: np.ndarray,
        mesh: Mesh,
        curl_ind: int = 0,
        derivative_boundary_constraints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes the curl_ind-th component of the curl of y at every point of
        the mesh.

        :param y: the values of y at every point of the mesh
        :param mesh: the mesh representing the discretized spatial domain
        :param curl_ind: the index of the component of the curl of y to
            compute; if y is a two dimensional vector field, it must be 0
        :param derivative_boundary_constraints: a 2D array (x dimension,
            y dimension) of boundary constraint pairs that allow for applying
            constraints to the calculated first derivatives before using them
            to compute the curl
        :return: the curl of y
        """
        self._verify_input_is_a_vector_field(y, mesh)
        if not (2 <= mesh.dimensions <= 3):
            raise ValueError(
                f"number of x dimensions ({mesh.dimensions}) must be 2 or 3"
            )
        if mesh.dimensions == 2 and curl_ind != 0:
            raise ValueError(f"curl index ({curl_ind}) must be 0 for 2D curl")
        if not (0 <= curl_ind < mesh.dimensions):
            raise ValueError(
                f"curl index ({curl_ind}) must be non-negative and less than "
                f"number of x dimensions ({mesh.dimensions})"
            )

        derivative_boundary_constraints = (
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints, mesh.dimensions, y.shape[-1]
            )
        )

        if mesh.coordinate_system_type == CoordinateSystem.CARTESIAN:
            if mesh.dimensions == 2 or curl_ind == 2:
                return self._derivative(
                    y[..., 1:2],
                    mesh.d_x[0],
                    0,
                    derivative_boundary_constraints[0, 1:2],
                ) - self._derivative(
                    y[..., :1],
                    mesh.d_x[1],
                    1,
                    derivative_boundary_constraints[1, :1],
                )

            elif curl_ind == 0:
                return self._derivative(
                    y[..., 2:],
                    mesh.d_x[1],
                    1,
                    derivative_boundary_constraints[1, 2:],
                ) - self._derivative(
                    y[..., 1:2],
                    mesh.d_x[2],
                    2,
                    derivative_boundary_constraints[2, 1:2],
                )

            else:
                return self._derivative(
                    y[..., :1],
                    mesh.d_x[2],
                    2,
                    derivative_boundary_constraints[2, :1],
                ) - self._derivative(
                    y[..., 2:],
                    mesh.d_x[0],
                    0,
                    derivative_boundary_constraints[0, 2:],
                )

        elif mesh.coordinate_system_type == CoordinateSystem.SPHERICAL:
            r = mesh.vertex_coordinate_grids[0][..., np.newaxis]

            if curl_ind == 0:
                phi = mesh.vertex_coordinate_grids[2][..., np.newaxis]
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)
                y_theta = y[..., 1:2]
                y_phi = y[..., 2:]
                d_y_theta_over_d_phi = self._derivative(
                    y_theta,
                    mesh.d_x[2],
                    2,
                    derivative_boundary_constraints[2, 1:2],
                )
                d_y_phi_over_d_theta = self._derivative(
                    y_phi,
                    mesh.d_x[1],
                    1,
                    derivative_boundary_constraints[1, 2:],
                )
                return (
                    d_y_theta_over_d_phi
                    + (cos_phi * y_theta - d_y_phi_over_d_theta) / sin_phi
                ) / r

            elif curl_ind == 1:
                y_r = y[..., :1]
                y_phi = y[..., 2:]
                d_y_r_over_d_phi = self._derivative(
                    y_r, mesh.d_x[2], 2, derivative_boundary_constraints[2, :1]
                )
                d_y_phi_over_d_r = self._derivative(
                    y_phi,
                    mesh.d_x[0],
                    0,
                    derivative_boundary_constraints[0, 2:],
                )
                return d_y_phi_over_d_r + (y_phi - d_y_r_over_d_phi) / r

            else:
                sin_phi = np.sin(
                    mesh.vertex_coordinate_grids[2][..., np.newaxis]
                )
                y_r = y[..., :1]
                y_theta = y[..., 1:2]
                d_y_r_over_d_theta = self._derivative(
                    y_r, mesh.d_x[1], 1, derivative_boundary_constraints[1, :1]
                )
                d_y_theta_over_d_r = self._derivative(
                    y_theta,
                    mesh.d_x[0],
                    0,
                    derivative_boundary_constraints[0, 1:2],
                )
                return (
                    -d_y_theta_over_d_r
                    + (d_y_r_over_d_theta / sin_phi - y_theta) / r
                )

        else:
            r = mesh.vertex_coordinate_grids[0][..., np.newaxis]

            if (
                mesh.coordinate_system_type == CoordinateSystem.POLAR
                or curl_ind == 2
            ):
                y_r = y[..., :1]
                y_theta = y[..., 1:2]
                d_y_r_over_d_theta = self._derivative(
                    y_r, mesh.d_x[1], 1, derivative_boundary_constraints[1, :1]
                )
                d_y_theta_over_d_r = self._derivative(
                    y_theta,
                    mesh.d_x[0],
                    0,
                    derivative_boundary_constraints[0, 1:2],
                )
                return d_y_theta_over_d_r + (y_theta - d_y_r_over_d_theta) / r

            elif curl_ind == 0:
                d_y_z_over_d_theta = self._derivative(
                    y[..., 2:],
                    mesh.d_x[1],
                    1,
                    derivative_boundary_constraints[1, 2:],
                )
                d_y_theta_over_d_z = self._derivative(
                    y[..., 1:2],
                    mesh.d_x[2],
                    2,
                    derivative_boundary_constraints[2, 1:2],
                )
                return d_y_z_over_d_theta / r - d_y_theta_over_d_z

            else:
                d_y_r_over_d_z = self._derivative(
                    y[..., :1],
                    mesh.d_x[2],
                    2,
                    derivative_boundary_constraints[2, :1],
                )
                d_y_z_over_d_r = self._derivative(
                    y[..., 2:],
                    mesh.d_x[0],
                    0,
                    derivative_boundary_constraints[0, 2:],
                )
                return d_y_r_over_d_z - d_y_z_over_d_r

    def laplacian(
        self,
        y: np.ndarray,
        mesh: Mesh,
        derivative_boundary_constraints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes the Laplacian of y at every point of the mesh.

        If the last rank of y has a dimension greater than one, the
        element-wise scalar Laplacian is computed instead of the vector
        Laplacian (which only makes a difference in curvilinear coordinate
        systems).

        :param y: the values of y at every point of the mesh
        :param mesh: the mesh representing the discretized spatial domain
        :param derivative_boundary_constraints: a 2D array (x dimension,
            y dimension) of boundary constraint pairs that allow for applying
            constraints to the calculated first derivatives before using them
            to compute the second derivatives and the Laplacian
        :return: the Laplacian of y
        """
        self._verify_input_shape_matches_mesh(y, mesh)

        derivative_boundary_constraints = (
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints, mesh.dimensions, y.shape[-1]
            )
        )

        if mesh.coordinate_system_type == CoordinateSystem.CARTESIAN:
            laplacian = np.zeros_like(y)
            for axis in range(y.ndim - 1):
                laplacian += self._second_derivative(
                    y,
                    mesh.d_x[axis],
                    mesh.d_x[axis],
                    axis,
                    axis,
                    derivative_boundary_constraints[axis, :],
                )

            return laplacian

        elif mesh.coordinate_system_type == CoordinateSystem.SPHERICAL:
            r = mesh.vertex_coordinate_grids[0][..., np.newaxis]
            phi = mesh.vertex_coordinate_grids[2][..., np.newaxis]
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            d_y_over_d_r = self._derivative(
                y, mesh.d_x[0], 0, derivative_boundary_constraints[0]
            )
            d_y_over_d_phi = self._derivative(
                y, mesh.d_x[2], 2, derivative_boundary_constraints[2]
            )
            d_sqr_y_over_d_r_sqr = self._second_derivative(
                y,
                mesh.d_x[0],
                mesh.d_x[0],
                0,
                0,
                derivative_boundary_constraints[0],
            )
            d_sqr_y_over_d_theta_sqr = self._second_derivative(
                y,
                mesh.d_x[1],
                mesh.d_x[1],
                1,
                1,
                derivative_boundary_constraints[1],
            )
            d_sqr_y_over_d_phi_sqr = self._second_derivative(
                y,
                mesh.d_x[2],
                mesh.d_x[2],
                2,
                2,
                derivative_boundary_constraints[2],
            )
            return (
                d_sqr_y_over_d_r_sqr
                + (
                    2 * d_y_over_d_r
                    + (
                        d_sqr_y_over_d_phi_sqr
                        + (
                            cos_phi * d_y_over_d_phi
                            + d_sqr_y_over_d_theta_sqr / sin_phi
                        )
                        / sin_phi
                    )
                    / r
                )
                / r
            )

        else:
            r = mesh.vertex_coordinate_grids[0][..., np.newaxis]
            d_y_over_d_r = self._derivative(
                y, mesh.d_x[0], 0, derivative_boundary_constraints[0]
            )
            d_sqr_y_over_d_r_sqr = self._second_derivative(
                y,
                mesh.d_x[0],
                mesh.d_x[0],
                0,
                0,
                derivative_boundary_constraints[0],
            )
            d_sqr_y_over_d_theta_sqr = self._second_derivative(
                y,
                mesh.d_x[1],
                mesh.d_x[1],
                1,
                1,
                derivative_boundary_constraints[1],
            )
            laplacian = (
                d_sqr_y_over_d_r_sqr
                + (d_sqr_y_over_d_theta_sqr / r + d_y_over_d_r) / r
            )

            if mesh.coordinate_system_type == CoordinateSystem.POLAR:
                return laplacian
            else:
                d_sqr_y_over_d_z_sqr = self._second_derivative(
                    y,
                    mesh.d_x[2],
                    mesh.d_x[2],
                    2,
                    2,
                    derivative_boundary_constraints[2],
                )
                return laplacian + d_sqr_y_over_d_z_sqr

    def vector_laplacian(
        self,
        y: np.ndarray,
        mesh: Mesh,
        vector_laplacian_ind: int,
        derivative_boundary_constraints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes the vector_laplacian_ind-th component of the vector Laplacian
        of y at every point of the mesh.

        :param y: the values of y at every point of the mesh
        :param mesh: the mesh representing the discretized spatial domain
        :param vector_laplacian_ind: the index of the component of the vector
            Laplacian of y to compute
        :param derivative_boundary_constraints: a 2D array (x dimension,
            y dimension) of boundary constraint pairs that allow for applying
            constraints to the calculated first derivatives before using them
            to compute the vector Laplacian
        :return: the vector Laplacian of y
        """
        self._verify_input_is_a_vector_field(y, mesh)
        if not (0 <= vector_laplacian_ind < mesh.dimensions):
            raise ValueError(
                f"vector Laplacian index ({vector_laplacian_ind}) must be "
                "non-negative and less than number of x dimensions "
                f"({mesh.dimensions})"
            )

        derivative_boundary_constraints = (
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints, mesh.dimensions, y.shape[-1]
            )
        )

        laplacian = self.laplacian(
            y[..., vector_laplacian_ind : vector_laplacian_ind + 1],
            mesh,
            derivative_boundary_constraints[
                :, vector_laplacian_ind : vector_laplacian_ind + 1
            ],
        )

        if mesh.coordinate_system_type == CoordinateSystem.CARTESIAN:
            return laplacian

        elif mesh.coordinate_system_type == CoordinateSystem.SPHERICAL:
            r = mesh.vertex_coordinate_grids[0][..., np.newaxis]
            phi = mesh.vertex_coordinate_grids[2][..., np.newaxis]
            y_r = y[..., :1]
            y_theta = y[..., 1:2]
            y_phi = y[..., 2:]
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            if vector_laplacian_ind == 1:
                d_y_theta_over_d_theta = self._derivative(
                    y_theta,
                    mesh.d_x[1],
                    1,
                    derivative_boundary_constraints[1, 1:2],
                )
                d_y_phi_over_d_phi = self._derivative(
                    y_phi,
                    mesh.d_x[2],
                    2,
                    derivative_boundary_constraints[2, 2:],
                )
                return (
                    laplacian
                    - 2.0
                    * (
                        y_r
                        + d_y_phi_over_d_phi
                        + (cos_phi * y_phi + d_y_theta_over_d_theta) / sin_phi
                    )
                    / r**2
                )

            elif vector_laplacian_ind == 2:
                d_y_r_over_d_theta = self._derivative(
                    y_r, mesh.d_x[1], 1, derivative_boundary_constraints[1, :1]
                )
                d_y_phi_over_d_theta = self._derivative(
                    y_phi,
                    mesh.d_x[1],
                    1,
                    derivative_boundary_constraints[1, 2:],
                )
                return laplacian + 2.0 * (
                    d_y_r_over_d_theta
                    + (cos_phi * d_y_phi_over_d_theta - y_theta / 2.0)
                    / sin_phi
                ) / (sin_phi * r**2)

            else:
                d_y_r_over_d_phi = self._derivative(
                    y_r, mesh.d_x[2], 2, derivative_boundary_constraints[2, :1]
                )
                d_y_theta_over_d_theta = self._derivative(
                    y_theta,
                    mesh.d_x[1],
                    1,
                    derivative_boundary_constraints[1, 1:2],
                )
                return (
                    laplacian
                    + 2.0
                    * (
                        d_y_r_over_d_phi
                        - (y_phi / 2.0 + cos_phi * d_y_theta_over_d_theta)
                        / sin_phi**2
                    )
                    / r**2
                )

        else:
            r = mesh.vertex_coordinate_grids[0][..., np.newaxis]

            if vector_laplacian_ind == 0:
                y_r = y[..., :1]
                y_theta = y[..., 1:2]
                d_y_theta_over_d_theta = self._derivative(
                    y_theta,
                    mesh.d_x[1],
                    1,
                    derivative_boundary_constraints[1, 1:2],
                )
                return (
                    laplacian - (y_r + 2.0 * d_y_theta_over_d_theta) / r**2
                )

            elif vector_laplacian_ind == 1:
                y_theta = y[..., 1:2]
                y_r = y[..., :1]
                d_y_r_over_d_theta = self._derivative(
                    y_r, mesh.d_x[1], 1, derivative_boundary_constraints[1, :1]
                )
                return (
                    laplacian - (y_theta - 2.0 * d_y_r_over_d_theta) / r**2
                )

            else:
                return laplacian

    def anti_laplacian(
        self,
        laplacian: np.ndarray,
        mesh: Mesh,
        y_constraints: Union[Sequence[Optional[Constraint]], np.ndarray],
        derivative_boundary_constraints: Optional[np.ndarray] = None,
        y_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes the inverse of the element-wise scalar Laplacian using the
        Jacobi method.

        :param laplacian: the right-hand side of the equation
        :param mesh: the mesh representing the discretized spatial domain
        :param y_constraints: a sequence of constraints on the values of the
            solution containing a constraint for each element of y; each
            constraint must constrain the boundary values of corresponding
            element of y for the system to be solvable
        :param derivative_boundary_constraints: an optional 2D array
            (x dimension, y dimension) of boundary constraint pairs that
            specify constraints on the first derivatives of the solution
        :param y_init: an optional initial estimate of the solution; if it is
            None, a random array is used
        :return: the array representing the solution to Poisson's equation at
            every point of the mesh
        """
        self._verify_input_shape_matches_mesh(laplacian, mesh, "Laplacian")

        derivative_boundary_constraints = (
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints,
                mesh.dimensions,
                laplacian.shape[-1],
            )
        )

        if y_init is None:
            y = np.random.random(laplacian.shape)
        else:
            if y_init.shape != laplacian.shape:
                raise ValueError
            y = y_init

        apply_constraints_along_last_axis(y_constraints, y)

        diff = np.inf
        while diff > self._tol:
            y_old = y
            y = self._next_anti_laplacian_estimate(
                y_old, laplacian, mesh, derivative_boundary_constraints
            )
            apply_constraints_along_last_axis(y_constraints, y)

            diff = float(np.linalg.norm(y - y_old))

        return y

    @staticmethod
    def _verify_input_shape_matches_mesh(
        input_array: np.ndarray, mesh: Mesh, input_name: str = "y"
    ):
        """
        Throws an error if the shape of the input array up to the last axis
        does not match the shape of the vertices of the mesh.

        :param input_array: the input array
        :param mesh: the mesh to compare against
        :param input_name: the name of the input array to include in the error
            message
        """
        if input_array.shape[:-1] != mesh.vertices_shape:
            raise ValueError(
                f"{input_name} shape up to second to last axis "
                f"{input_array.shape[:-1]} must match mesh vertices shape "
                f"{mesh.vertices_shape}"
            )

    @staticmethod
    def _verify_input_is_a_vector_field(input_array: np.ndarray, mesh: Mesh):
        """
        Throws an error if the shape of the input array is not that of a vector
        field evaluated over the vertices of the provided mesh.

        :param input_array: the input array
        :param mesh: the mesh to compare against
        """
        NumericalDifferentiator._verify_input_shape_matches_mesh(
            input_array, mesh
        )
        if input_array.shape[-1] != mesh.dimensions:
            raise ValueError(
                f"y value vector length ({input_array.shape[-1]}) "
                f"must match number of x dimensions ({mesh.dimensions})"
            )

    @staticmethod
    def _verify_and_get_derivative_boundary_constraints(
        derivative_boundary_constraints: Optional[np.ndarray],
        x_axes: int,
        y_elements: int,
    ) -> np.ndarray:
        """
        If the provided derivative boundary constraints are not None, it just
        asserts that the input array's shape matches expectations and returns
        it. Otherwise it creates an array of empty objects of the correct
        shape and returns that.

        :param derivative_boundary_constraints: an array of derivative boundary
            constraint pairs
        :param x_axes: the number of spatial dimensions
        :param y_elements: the number of elements y has
        :return: an array of derivative boundary constraint pairs or empty
            objects depending on whether the input array is None
        """
        if derivative_boundary_constraints is None:
            return np.empty((x_axes, y_elements), dtype=object)

        if derivative_boundary_constraints.shape != (x_axes, y_elements):
            raise ValueError(
                "expected derivative boundary constraints shape to be "
                f"{(x_axes, y_elements)} but got "
                f"{derivative_boundary_constraints.shape}"
            )

        return derivative_boundary_constraints


class ThreePointCentralDifferenceMethod(NumericalDifferentiator):
    """
    A numerical differentiator using a three-point (second order) central
    difference approximation.
    """

    def __init__(self, tol: float = 1e-3):
        """
        :param tol: the stopping criterion for the Jacobi algorithm when
            computing the anti-Laplacian
        """
        super(ThreePointCentralDifferenceMethod, self).__init__(tol)

    def _derivative(
        self,
        y: np.ndarray,
        d_x: float,
        x_axis: int,
        derivative_boundary_constraints: Union[
            Sequence[Optional[BoundaryConstraintPair]], np.ndarray
        ],
    ) -> np.ndarray:
        if y.shape[x_axis] <= 2:
            raise ValueError(
                f"y must contain at least 3 points along x-axis ({x_axis})"
            )

        slicer: Slicer = [slice(None)] * y.ndim
        halo = np.zeros(y.shape[:x_axis] + (1,) + y.shape[x_axis + 1 :])
        y_extended = np.concatenate([halo, y, halo], axis=x_axis)

        slicer[x_axis] = slice(0, -2)
        y_prev = y_extended[tuple(slicer)]
        slicer[x_axis] = slice(2, None)
        y_next = y_extended[tuple(slicer)]

        derivative = (y_next - y_prev) / (2.0 * d_x)

        slicer[x_axis] = slice(None)

        for y_ind, constraint_pair in enumerate(
            derivative_boundary_constraints
        ):
            if constraint_pair is None:
                continue

            slicer[-1] = slice(y_ind, y_ind + 1)

            lower_boundary_constraint = constraint_pair[0]
            if lower_boundary_constraint is not None:
                slicer[x_axis] = slice(0, 1)
                lower_boundary_constraint.apply(derivative[tuple(slicer)])

            upper_boundary_constraint = constraint_pair[1]
            if upper_boundary_constraint is not None:
                slicer[x_axis] = slice(-1, None)
                upper_boundary_constraint.apply(derivative[tuple(slicer)])

        return derivative

    def _second_derivative(
        self,
        y: np.ndarray,
        d_x1: float,
        d_x2: float,
        x_axis1: int,
        x_axis2: int,
        derivative_boundary_constraints: Union[
            Sequence[Optional[BoundaryConstraintPair]], np.ndarray
        ],
    ) -> np.ndarray:
        if x_axis1 != x_axis2:
            first_derivative = self._derivative(
                y, d_x1, x_axis1, derivative_boundary_constraints
            )
            return self._derivative(
                first_derivative, d_x2, x_axis2, [None] * y.shape[-1]
            )

        if y.shape[x_axis1] <= 2:
            raise ValueError(
                f"y must contain at least 3 points along x-axis ({x_axis1})"
            )

        slicer: Slicer = [slice(None)] * y.ndim
        y_extended = self._add_halos_along_axis(
            y, x_axis1, d_x1, slicer, derivative_boundary_constraints
        )

        slicer[x_axis1] = slice(0, -2)
        y_prev = y_extended[tuple(slicer)]
        slicer[x_axis1] = slice(1, -1)
        y_curr = y_extended[tuple(slicer)]
        slicer[x_axis1] = slice(2, None)
        y_next = y_extended[tuple(slicer)]

        return (y_next - 2.0 * y_curr + y_prev) / (d_x1 * d_x2)

    def _next_anti_laplacian_estimate(
        self,
        y_hat: np.ndarray,
        laplacian: np.ndarray,
        mesh: Mesh,
        derivative_boundary_constraints: Optional[np.ndarray],
    ) -> np.ndarray:
        if not np.all(np.array(y_hat.shape[:-1]) > 2):
            raise ValueError(
                "y must contain at least 3 points along all x axes"
            )

        slicer: Slicer = [slice(None)] * y_hat.ndim
        anti_laplacian = np.zeros_like(y_hat)

        all_d_x_sqr = np.square(mesh.d_x)
        r = r_sqr = phi = sin_phi = r_sqr_sin_phi_sqr = None
        if mesh.coordinate_system_type != CoordinateSystem.CARTESIAN:
            r = mesh.vertex_coordinate_grids[0][..., np.newaxis]
            r_sqr = r**2

            if mesh.coordinate_system_type == CoordinateSystem.SPHERICAL:
                phi = mesh.vertex_coordinate_grids[2][..., np.newaxis]
                sin_phi = np.sin(phi)
                r_sqr_sin_phi_sqr = r_sqr * sin_phi**2

        for axis, d_x in enumerate(mesh.d_x):
            d_x_sqr = all_d_x_sqr[axis]
            y_hat_extended = self._add_halos_along_axis(
                y_hat, axis, d_x, slicer, derivative_boundary_constraints[axis]
            )

            slicer[axis] = slice(0, -2)
            y_hat_prev = y_hat_extended[tuple(slicer)]
            slicer[axis] = slice(2, None)
            y_hat_next = y_hat_extended[tuple(slicer)]

            anti_laplacian_update = (y_hat_prev + y_hat_next) / d_x_sqr

            if mesh.coordinate_system_type == CoordinateSystem.CARTESIAN:
                anti_laplacian += anti_laplacian_update

            elif mesh.coordinate_system_type == CoordinateSystem.SPHERICAL:
                if axis == 0:
                    anti_laplacian += anti_laplacian_update + (
                        y_hat_next - y_hat_prev
                    ) / (d_x * r)
                elif axis == 1:
                    anti_laplacian += anti_laplacian_update / r_sqr_sin_phi_sqr
                else:
                    anti_laplacian += (
                        anti_laplacian_update
                        + np.cos(phi)
                        * (y_hat_next - y_hat_prev)
                        / (2.0 * d_x * sin_phi)
                    ) / r_sqr

            else:
                if axis == 0:
                    anti_laplacian += anti_laplacian_update + (
                        y_hat_next - y_hat_prev
                    ) / (2.0 * d_x * r)
                elif axis == 1:
                    anti_laplacian += anti_laplacian_update / r_sqr
                else:
                    anti_laplacian += anti_laplacian_update

            slicer[axis] = slice(None)

        anti_laplacian -= laplacian

        if mesh.coordinate_system_type == CoordinateSystem.CARTESIAN:
            return anti_laplacian / (2.0 / all_d_x_sqr).sum()

        elif mesh.coordinate_system_type == CoordinateSystem.SPHERICAL:
            return anti_laplacian / (
                2.0 / all_d_x_sqr[0]
                + 2.0 / (all_d_x_sqr[1] * r_sqr_sin_phi_sqr)
                + 2.0 / (all_d_x_sqr[2] * r_sqr)
            )

        else:
            step_size_coefficient = 2.0 / all_d_x_sqr[0] + 2.0 / (
                all_d_x_sqr[1] * r_sqr
            )
            if mesh.coordinate_system_type == CoordinateSystem.POLAR:
                return anti_laplacian / step_size_coefficient
            else:
                step_size_coefficient += 2.0 / all_d_x_sqr[2]
                return anti_laplacian / step_size_coefficient

    @staticmethod
    def _add_halos_along_axis(
        y: np.ndarray,
        x_axis: int,
        d_x: float,
        slicer: Slicer,
        derivative_boundary_constraints: Union[
            Sequence[Optional[BoundaryConstraintPair]], np.ndarray
        ],
    ) -> np.ndarray:
        """
        Adds halo vertices to y along the specified axis based on the provided
        derivative boundary constraints and the values at the vertices adjacent
        to the boundary vertices.

        :param y: the input array to extend with halos
        :param x_axis: the axis along which the halos are to be added
        :param d_x: the spatial step size
        :param slicer: the slicer array to use for indexing the vertices
            adjacent to the boundaries
        :param derivative_boundary_constraints: the derivative boundary
            constraints
        :return: y extended with halo vertices along the specified axis
        """
        slicer[x_axis] = slice(1, 2)
        y_lower_boundary_adjacent = y[tuple(slicer)]
        slicer[x_axis] = slice(-2, -1)
        y_upper_boundary_adjacent = y[tuple(slicer)]

        y_lower_halo = np.zeros_like(y_lower_boundary_adjacent)
        y_upper_halo = np.zeros_like(y_upper_boundary_adjacent)

        for y_ind, boundary_constraint_pair in enumerate(
            derivative_boundary_constraints
        ):
            if boundary_constraint_pair is None:
                continue

            lower_boundary_constraint = boundary_constraint_pair[0]
            if lower_boundary_constraint is not None:
                lower_boundary_constraint.multiply_and_add(
                    y_lower_boundary_adjacent[..., y_ind : y_ind + 1],
                    -2.0 * d_x,
                    y_lower_halo[..., y_ind : y_ind + 1],
                )

            upper_boundary_constraint = boundary_constraint_pair[1]
            if upper_boundary_constraint is not None:
                upper_boundary_constraint.multiply_and_add(
                    y_upper_boundary_adjacent[..., y_ind : y_ind + 1],
                    2.0 * d_x,
                    y_upper_halo[..., y_ind : y_ind + 1],
                )

        return np.concatenate([y_lower_halo, y, y_upper_halo], axis=x_axis)
