from copy import deepcopy

def solve_system(matrix):
    mat = [row[:] for row in matrix]
    n = len(mat)

    for i in range(n):
        max_row = i
        max_val = abs(mat[i][i])
        for r in range(i+1, n):
            if abs(mat[r][i]) > max_val:
                max_val = abs(mat[r][i])
                max_row = r
        if max_row != i:
            mat[i], mat[max_row] = mat[max_row], mat[i]

        pivot = mat[i][i]
        if abs(pivot) < 1e-14:
            raise ValueError("Matrix is singular or nearly singular.")

        for r in range(i+1, n):
            factor = mat[r][i] / pivot
            for c in range(i, n+1):
                mat[r][c] -= factor * mat[i][c]

    result = [0] * n
    for i in reversed(range(n)):
        total = 0
        for j in range(i+1, n):
            total += mat[i][j] * result[j]
        result[i] = (mat[i][n] - total) / mat[i][i]

    return result

def lu_decompose(matrix):
    n = len(matrix)
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0

    for i in range(n):
        for j in range(i, n):
            sum_ = 0.0
            for k in range(i):
                sum_ += L[i][k] * U[k][j]
            U[i][j] = matrix[i][j] - sum_

        for j in range(i+1, n):
            sum_ = 0.0
            for k in range(i):
                sum_ += L[j][k] * U[k][i]
            if abs(U[i][i]) < 1e-14:
                L[j][i] = 0.0
            else:
                L[j][i] = (matrix[j][i] - sum_) / U[i][i]

    det = 1.0
    for i in range(n):
        det *= U[i][i]

    return L, U, det

def check_diagonal_dominance(matrix):
    n = len(matrix)
    for i in range(n):
        diagonal = abs(matrix[i][i])
        off_diag_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        if diagonal <= off_diag_sum:
            return False
    return True

def check_positive_definite(matrix):
    n = len(matrix)

    def submatrix_det(k):
        sub = [row[:k] for row in matrix[:k]]
        return compute_determinant(sub)

    def compute_determinant(mat):
        mat_copy = deepcopy(mat)
        size = len(mat_copy)
        det_value = 1.0
        for i in range(size):
            pivot = mat_copy[i][i]
            if abs(pivot) < 1e-14:
                return 0.0
            det_value *= pivot
            for r in range(i+1, size):
                factor = mat_copy[r][i] / pivot
                for c in range(i, size):
                    mat_copy[r][c] -= factor * mat_copy[i][c]
        return det_value

    for k in range(1, n+1):
        if submatrix_det(k) <= 0:
            return False
    return True

def run_example():
    matrix1 = [
        [2, -1, 1, 6],
        [1, 3, 1, 0],
        [-1, 5, 4, -3]
    ]
    solution1 = solve_system(matrix1)
    print("1) Solution to the system:")
    for i, val in enumerate(solution1):
        print(f"   x{i+1} =", val)
    print()

    matrix2 = [
        [1, 1, 0, 3],
        [2, 1, -1, 1],
        [3, -1, -1, 2],
        [-1, 2, 3, -1]
    ]
    L2, U2, det2 = lu_decompose(matrix2)
    print("2) LU Decomposition Results:")
    print("   Determinant:", det2)
    print("   L matrix:")
    for row in L2:
        print("     ", [int(round(x)) for x in row])
    print("   U matrix:")
    for row in U2:
        print("     ", [int(round(x)) for x in row])
    print()

    matrix3 = [
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ]
    dd3 = check_diagonal_dominance(matrix3)
    print("3) Is the matrix diagonally dominant?", dd3)
    print()

    matrix4 = [
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ]
    pd4 = check_positive_definite(matrix4)
    print("4) Is the matrix positive definite?", pd4)
    print()

if __name__ == "__main__":
    run_example()

