import numpy as np


def compute_displacement(abs_pos, L, Ng, order='F'):
    # Memory layout of multi-dimensional arrays (row vs column major)
    displacement = np.zeros(np.shape(abs_pos))
    initial_pos = np.zeros(np.shape(abs_pos))
    dx = L / Ng

    for i in range(0, Ng):
        for j in range(0, Ng):
            for k in range(0, Ng):
                if order == 'F':
                    n = k + Ng * (j + Ng * i)  # column
                elif order == 'C':
                    n = i + Ng * (j + Ng * k)  # row

                qx = i * dx
                qy = j * dx
                qz = k * dx

                initial_pos[n] = [qx, qy, qz]

                displacement[n] = abs_pos[n] - [qx, qy, qz]

    return initial_pos, displacement


def compute_cic(x_in, y_in, z_in, boxsize, Ngrid):
    # TODO: implement in JAX for autodiff
    cell_len = np.float(boxsize) / np.float(Ngrid)
    print('cell_len = ', cell_len)

    x_dat = x_in / cell_len
    y_dat = y_in / cell_len
    z_dat = z_in / cell_len

    # Create a new grid which will contain the densities
    grid = np.zeros([Ngrid, Ngrid, Ngrid], dtype='float64')

    # Find cell center coordinates
    x_c = np.floor(x_dat).astype(int)
    y_c = np.floor(y_dat).astype(int)
    z_c = np.floor(z_dat).astype(int)

    # Calculating contributions for the CIC interpolation
    d_x = x_dat - x_c
    d_y = y_dat - y_c
    d_z = z_dat - z_c

    t_x = 1. - d_x
    t_y = 1. - d_y
    t_z = 1. - d_z

    # Enforce periodicity for cell center coordinates + 1
    X = (x_c + 1) % Ngrid
    Y = (y_c + 1) % Ngrid
    Z = (z_c + 1) % Ngrid

    # Populate the density grid according to the CIC scheme

    aux, edges = np.histogramdd(np.array([z_c, y_c, x_c]).T, weights=t_x * t_y * t_z, bins=(Ngrid, Ngrid, Ngrid),
                                range=[[0, Ngrid], [0, Ngrid], [0, Ngrid]])
    grid += aux

    aux, edges = np.histogramdd(np.array([z_c, y_c, X]).T, weights=d_x * t_y * t_z, bins=(Ngrid, Ngrid, Ngrid),
                                range=[[0, Ngrid], [0, Ngrid], [0, Ngrid]])
    grid += aux

    aux, edges = np.histogramdd(np.array([z_c, Y, x_c]).T, weights=t_x * d_y * t_z, bins=(Ngrid, Ngrid, Ngrid),
                                range=[[0, Ngrid], [0, Ngrid], [0, Ngrid]])
    grid += aux

    aux, edges = np.histogramdd(np.array([Z, y_c, x_c]).T, weights=t_x * t_y * d_z, bins=(Ngrid, Ngrid, Ngrid),
                                range=[[0, Ngrid], [0, Ngrid], [0, Ngrid]])
    grid += aux

    aux, edges = np.histogramdd(np.array([z_c, Y, X]).T, weights=d_x * d_y * t_z, bins=(Ngrid, Ngrid, Ngrid),
                                range=[[0, Ngrid], [0, Ngrid], [0, Ngrid]])
    grid += aux

    aux, edges = np.histogramdd(np.array([Z, Y, x_c]).T, weights=t_x * d_y * d_z, bins=(Ngrid, Ngrid, Ngrid),
                                range=[[0, Ngrid], [0, Ngrid], [0, Ngrid]])
    grid += aux

    aux, edges = np.histogramdd(np.array([Z, y_c, X]).T, weights=d_x * t_y * d_z, bins=(Ngrid, Ngrid, Ngrid),
                                range=[[0, Ngrid], [0, Ngrid], [0, Ngrid]])
    grid += aux

    aux, edges = np.histogramdd(np.array([Z, Y, X]).T, weights=d_x * d_y * d_z, bins=(Ngrid, Ngrid, Ngrid),
                                range=[[0, Ngrid], [0, Ngrid], [0, Ngrid]])
    grid += aux

    return grid
