#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>

using namespace std;

// =====================
// CSR Sparse Matrix
// =====================
struct SparseMatrixCSR
{
    int n;

    vector<double> values;

    vector<int> col_idx;

    vector<int> row_ptr;

    vector<double> diag;
};

// =====================
// Generate Sparse Matrix
// Strictly Diagonally Dominant
// =====================
SparseMatrixCSR generate_sparse_matrix(
    int n,
    double density)
{
    SparseMatrixCSR A;

    A.n = n;

    A.row_ptr.resize(n + 1);

    A.diag.resize(n);

    random_device rd;

    mt19937 gen(rd());

    uniform_real_distribution<double> val_dist(-1.0, 1.0);

    int nnz_per_row =
        max(1, (int)(n * density));

    vector<vector<pair<int, double>>> rows(n);

    for (int i = 0; i < n; ++i)
    {
        vector<int> cols;

        while ((int)cols.size() < nnz_per_row)
        {
            int j = gen() % n;

            if (j == i)
                continue;

            bool exist = false;

            for (int c : cols)
            {
                if (c == j)
                {
                    exist = true;
                    break;
                }
            }

            if (!exist)
                cols.push_back(j);
        }

        double row_sum = 0.0;

        for (int j : cols)
        {
            double val = val_dist(gen);

            rows[i].push_back({ j, val });

            row_sum += abs(val);
        }

        // strict diagonal dominance
        double diag_val = row_sum + 1.0;

        rows[i].push_back({ i, diag_val });

        A.diag[i] = diag_val;
    }

    // Build CSR
    A.row_ptr[0] = 0;

    for (int i = 0; i < n; ++i)
    {
        sort(rows[i].begin(), rows[i].end());

        for (auto& entry : rows[i])
        {
            A.values.push_back(entry.second);

            A.col_idx.push_back(entry.first);
        }

        A.row_ptr[i + 1] =
            A.values.size();
    }

    return A;
}

// =====================
// Sparse Matrix Vector Multiply
// y = A*x
// =====================
void spmv(
    const SparseMatrixCSR& A,
    const vector<double>& x,
    vector<double>& y)
{
    int n = A.n;

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        double sum = 0.0;

        for (int k = A.row_ptr[i];
            k < A.row_ptr[i + 1];
            ++k)
        {
            sum +=
                A.values[k] *
                x[A.col_idx[k]];
        }

        y[i] = sum;
    }
}

// =====================
// Jacobi Iteration
// Parallel
// =====================
void jacobi(
    const SparseMatrixCSR& A,
    const vector<double>& b,
    vector<double>& x,
    int max_iter,
    double tol,
    int& iter_out,
    double& error_out)
{
    int n = A.n;

    vector<double> x_new(n);

    double start =
        omp_get_wtime();

    for (int iter = 0;
        iter < max_iter;
        ++iter)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            double sigma = 0.0;

            for (int k = A.row_ptr[i];
                k < A.row_ptr[i + 1];
                ++k)
            {
                int j = A.col_idx[k];

                if (j != i)
                {
                    sigma +=
                        A.values[k] * x[j];
                }
            }

            x_new[i] =
                (b[i] - sigma)
                / A.diag[i];
        }

        double max_diff = 0.0;

#pragma omp parallel for reduction(max:max_diff)
        for (int i = 0; i < n; ++i)
        {
            double diff =
                abs(x_new[i] - x[i]);

            if (diff > max_diff)
                max_diff = diff;

            x[i] = x_new[i];
        }

        if (max_diff < tol)
        {
            iter_out = iter + 1;

            error_out = max_diff;

            double end =
                omp_get_wtime();

            cout << "Jacobi time: "
                << end - start
                << " s\n";

            return;
        }
    }

    iter_out = max_iter;

    error_out = -1.0;
}

// =====================
// Gauss-Seidel
// Serial
// =====================
void gauss_seidel(
    const SparseMatrixCSR& A,
    const vector<double>& b,
    vector<double>& x,
    int max_iter,
    double tol,
    int& iter_out,
    double& error_out)
{
    int n = A.n;

    double start =
        omp_get_wtime();

    for (int iter = 0;
        iter < max_iter;
        ++iter)
    {
        double max_diff = 0.0;

        for (int i = 0; i < n; ++i)
        {
            double sigma = 0.0;

            for (int k = A.row_ptr[i];
                k < A.row_ptr[i + 1];
                ++k)
            {
                int j = A.col_idx[k];

                if (j != i)
                {
                    sigma +=
                        A.values[k] * x[j];
                }
            }

            double x_new =
                (b[i] - sigma)
                / A.diag[i];

            double diff =
                abs(x_new - x[i]);

            if (diff > max_diff)
                max_diff = diff;

            x[i] = x_new;
        }

        if (max_diff < tol)
        {
            iter_out = iter + 1;

            error_out = max_diff;

            double end =
                omp_get_wtime();

            cout << "Gauss-Seidel time: "
                << end - start
                << " s\n";

            return;
        }
    }

    iter_out = max_iter;

    error_out = -1.0;
}

// =====================
// Compute L2 Error
// =====================
double compute_error(
    const vector<double>& x_true,
    const vector<double>& x)
{
    double err = 0.0;

#pragma omp parallel for reduction(+:err)
    for (int i = 0;
        i < x.size();
        ++i)
    {
        double diff =
            x_true[i] - x[i];

        err += diff * diff;
    }

    return sqrt(err);
}

// =====================
// Main
// =====================
int main()
{
    const int N = 1000;

    const double DENSITY = 0.01;

    const int MAX_ITER = 5000;

    const double TOL = 1e-8;

    omp_set_num_threads(8);

    cout << "Generating sparse matrix...\n";

    SparseMatrixCSR A =
        generate_sparse_matrix(
            N,
            DENSITY);

    // true solution
    vector<double> b_true(N);

    random_device rd;

    mt19937 gen(rd());

    uniform_real_distribution<double>
        dist(0.0, 1.0);

#pragma omp parallel
    {
        mt19937 local_gen(rd() + omp_get_thread_num());

#pragma omp for
        for (int i = 0; i < N; ++i)
        {
            b_true[i] =
                dist(local_gen);
        }
    }

    // c = A*b
    vector<double> c(N);

    spmv(A, b_true, c);

    // =====================
    // Jacobi
    // =====================
    cout << "Running Jacobi...\n";

    vector<double> x_jacobi(
        N,
        0.0);

    int iter_j;

    double err_j;

    jacobi(
        A,
        c,
        x_jacobi,
        MAX_ITER,
        TOL,
        iter_j,
        err_j);

    cout << "Jacobi iterations: "
        << iter_j << endl;

    cout << "Jacobi residual: "
        << err_j << endl;

    cout << "Jacobi solution error: "
        << compute_error(
            b_true,
            x_jacobi)
        << endl;

    // =====================
    // Gauss-Seidel
    // =====================
    cout << "\nRunning Gauss-Seidel...\n";

    vector<double> x_gs(
        N,
        0.0);

    int iter_gs;

    double err_gs;

    gauss_seidel(
        A,
        c,
        x_gs,
        MAX_ITER,
        TOL,
        iter_gs,
        err_gs);

    cout << "Gauss-Seidel iterations: "
        << iter_gs << endl;

    cout << "Gauss-Seidel residual: "
        << err_gs << endl;

    cout << "Gauss-Seidel solution error: "
        << compute_error(
            b_true,
            x_gs)
        << endl;

    return 0;
}