import numpy as np


EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
CASES = {
    0: (), 1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),),
    4: ((1, 2),), 6: ((0, 2),), 7: ((3, 2),), 8: ((2, 3),),
    9: ((2, 0),), 11: ((1, 2),), 12: ((1, 3),),
    13: ((0, 1),), 14: ((3, 0),), 15: (),
}
PAIRINGS = {"A": ((0, 1), (2, 3)), "B": ((3, 0), (1, 2))}


def bilinear_coefficients(values):
    f0, f1, f2, f3 = values
    return f0, f1 - f0, f3 - f0, (f2 - f1) - (f3 - f0)


def edge_crossings(corners, values):
    crossings = {}
    for edge, (start, end) in enumerate(EDGES):
        f0, f1 = values[start], values[end]
        if (f0 < 0) != (f1 < 0):
            t = -f0 / (f1 - f0)
            crossings[edge] = (1 - t) * corners[start] + t * corners[end]
    return crossings


def asymptotic_decider(values):
    f0, f1, f2, f3 = values
    product_02, product_13 = f0 * f2, f1 * f3
    determinant = product_02 - product_13
    tolerance = 4 * np.finfo(float).eps * max(abs(product_02), abs(product_13))
    if abs(determinant) <= tolerance:
        return "saddle"
    return "A" if determinant > 0 else "B"


def cell_segments(corners, values, level=0):
    """Zwraca odcinki izokonturu w jednej komórce."""
    values = np.asarray(values, dtype=float) - level
    mask = sum(1 << i for i, value in enumerate(values) if value < 0)
    if mask in (0, 15):
        return []

    crossings = edge_crossings(corners, values)
    if mask in (5, 10):
        variant = asymptotic_decider(values)
        if variant == "saddle":
            _, b, c, d = bilinear_coefficients(values)
            u, v = -c / d, -b / d
            saddle = corners[0] + u * (corners[1] - corners[0]) + v * (
                corners[3] - corners[0]
            )
            return [np.array([point, saddle]) for point in crossings.values()]
        pairs = PAIRINGS[variant]
    else:
        pairs = CASES[mask]
    return [
        np.array([crossings[start], crossings[end]])
        for start, end in pairs
        if not np.array_equal(crossings[start], crossings[end])
    ]
