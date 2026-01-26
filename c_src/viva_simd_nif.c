/**
 * viva_simd_nif.c - SIMD-optimized operations for VIVA neural networks
 *
 * Uses AVX/AVX2/SSE for vectorized operations on RTX 4090 platform.
 * Falls back to scalar operations when SIMD not available.
 *
 * Compile with:
 *   gcc -O3 -mavx2 -mfma -fPIC -shared -o priv/viva_simd_nif.so c_src/viva_simd_nif.c -I$(ERL_INCLUDE_PATH)
 */

#include <erl_nif.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* Check for AVX support */
#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#define HAS_AVX 1
#else
#define HAS_AVX 0
#endif

/* ============================================================================
 * SIMD DOT PRODUCT
 * ============================================================================ */

#if HAS_AVX
/**
 * AVX-optimized dot product for aligned data
 * Processes 8 floats at a time (256-bit registers)
 */
static double dot_product_avx(const double* a, const double* b, size_t n) {
    __m256d sum = _mm256_setzero_pd();
    size_t i = 0;

    /* Process 4 doubles at a time with AVX */
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        sum = _mm256_add_pd(sum, _mm256_mul_pd(va, vb));
    }

    /* Horizontal sum of the 4 doubles in sum */
    __m128d low = _mm256_castpd256_pd128(sum);
    __m128d high = _mm256_extractf128_pd(sum, 1);
    __m128d sum128 = _mm_add_pd(low, high);
    sum128 = _mm_hadd_pd(sum128, sum128);

    double result;
    _mm_store_sd(&result, sum128);

    /* Handle remaining elements */
    for (; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}
#endif

/**
 * Scalar fallback dot product
 */
static double dot_product_scalar(const double* a, const double* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 * NIF: simd_dot(ListA, ListB) -> Float
 */
static ERL_NIF_TERM simd_dot(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    unsigned int len_a, len_b;

    if (!enif_get_list_length(env, argv[0], &len_a) ||
        !enif_get_list_length(env, argv[1], &len_b) ||
        len_a != len_b || len_a == 0) {
        return enif_make_badarg(env);
    }

    /* Allocate arrays */
    double* a = (double*)enif_alloc(len_a * sizeof(double));
    double* b = (double*)enif_alloc(len_b * sizeof(double));

    if (!a || !b) {
        if (a) enif_free(a);
        if (b) enif_free(b);
        return enif_make_badarg(env);
    }

    /* Convert Erlang lists to C arrays */
    ERL_NIF_TERM head, tail_a = argv[0], tail_b = argv[1];
    for (unsigned int i = 0; i < len_a; i++) {
        enif_get_list_cell(env, tail_a, &head, &tail_a);
        enif_get_double(env, head, &a[i]);
        enif_get_list_cell(env, tail_b, &head, &tail_b);
        enif_get_double(env, head, &b[i]);
    }

    /* Compute dot product */
    double result;
#if HAS_AVX
    result = dot_product_avx(a, b, len_a);
#else
    result = dot_product_scalar(a, b, len_a);
#endif

    enif_free(a);
    enif_free(b);

    return enif_make_double(env, result);
}

/* ============================================================================
 * SIMD ELEMENT-WISE MULTIPLY
 * ============================================================================ */

#if HAS_AVX
/**
 * AVX-optimized element-wise multiply
 */
static void mul_avx(const double* a, const double* b, double* c, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vc = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(&c[i], vc);
    }
    for (; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}
#endif

static void mul_scalar(const double* a, const double* b, double* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

/**
 * NIF: simd_mul(ListA, ListB) -> List
 */
static ERL_NIF_TERM simd_mul(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    unsigned int len_a, len_b;

    if (!enif_get_list_length(env, argv[0], &len_a) ||
        !enif_get_list_length(env, argv[1], &len_b) ||
        len_a != len_b || len_a == 0) {
        return enif_make_badarg(env);
    }

    double* a = (double*)enif_alloc(len_a * sizeof(double));
    double* b = (double*)enif_alloc(len_a * sizeof(double));
    double* c = (double*)enif_alloc(len_a * sizeof(double));

    if (!a || !b || !c) {
        if (a) enif_free(a);
        if (b) enif_free(b);
        if (c) enif_free(c);
        return enif_make_badarg(env);
    }

    ERL_NIF_TERM head, tail_a = argv[0], tail_b = argv[1];
    for (unsigned int i = 0; i < len_a; i++) {
        enif_get_list_cell(env, tail_a, &head, &tail_a);
        enif_get_double(env, head, &a[i]);
        enif_get_list_cell(env, tail_b, &head, &tail_b);
        enif_get_double(env, head, &b[i]);
    }

#if HAS_AVX
    mul_avx(a, b, c, len_a);
#else
    mul_scalar(a, b, c, len_a);
#endif

    /* Convert back to Erlang list */
    ERL_NIF_TERM* terms = (ERL_NIF_TERM*)enif_alloc(len_a * sizeof(ERL_NIF_TERM));
    for (unsigned int i = 0; i < len_a; i++) {
        terms[i] = enif_make_double(env, c[i]);
    }
    ERL_NIF_TERM result = enif_make_list_from_array(env, terms, len_a);

    enif_free(a);
    enif_free(b);
    enif_free(c);
    enif_free(terms);

    return result;
}

/* ============================================================================
 * SIMD MATRIX MULTIPLY
 * ============================================================================ */

/**
 * NIF: simd_matmul(A, B, M, K, N) -> List
 * A is MxK, B is KxN, result is MxN
 */
static ERL_NIF_TERM simd_matmul(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    unsigned int len_a, len_b;
    int M, K, N;

    if (!enif_get_list_length(env, argv[0], &len_a) ||
        !enif_get_list_length(env, argv[1], &len_b) ||
        !enif_get_int(env, argv[2], &M) ||
        !enif_get_int(env, argv[3], &K) ||
        !enif_get_int(env, argv[4], &N)) {
        return enif_make_badarg(env);
    }

    if ((unsigned int)(M * K) != len_a || (unsigned int)(K * N) != len_b) {
        return enif_make_badarg(env);
    }

    double* A = (double*)enif_alloc(len_a * sizeof(double));
    double* B = (double*)enif_alloc(len_b * sizeof(double));
    double* C = (double*)enif_alloc(M * N * sizeof(double));

    if (!A || !B || !C) {
        if (A) enif_free(A);
        if (B) enif_free(B);
        if (C) enif_free(C);
        return enif_make_badarg(env);
    }

    /* Load matrices */
    ERL_NIF_TERM head, tail = argv[0];
    for (unsigned int i = 0; i < len_a; i++) {
        enif_get_list_cell(env, tail, &head, &tail);
        enif_get_double(env, head, &A[i]);
    }
    tail = argv[1];
    for (unsigned int i = 0; i < len_b; i++) {
        enif_get_list_cell(env, tail, &head, &tail);
        enif_get_double(env, head, &B[i]);
    }

    /* Matrix multiply with optional SIMD for inner loop */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
#if HAS_AVX
            /* Use SIMD for inner product */
            __m256d vsum = _mm256_setzero_pd();
            int k = 0;
            for (; k + 4 <= K; k += 4) {
                __m256d va = _mm256_loadu_pd(&A[i * K + k]);
                /* Gather B elements (not contiguous) */
                __m256d vb = _mm256_set_pd(
                    B[(k + 3) * N + j],
                    B[(k + 2) * N + j],
                    B[(k + 1) * N + j],
                    B[k * N + j]
                );
                vsum = _mm256_add_pd(vsum, _mm256_mul_pd(va, vb));
            }
            /* Horizontal sum */
            __m128d low = _mm256_castpd256_pd128(vsum);
            __m128d high = _mm256_extractf128_pd(vsum, 1);
            __m128d sum128 = _mm_add_pd(low, high);
            sum128 = _mm_hadd_pd(sum128, sum128);
            _mm_store_sd(&sum, sum128);
            /* Remainder */
            for (; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
#else
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
#endif
            C[i * N + j] = sum;
        }
    }

    /* Convert result to list */
    unsigned int result_len = M * N;
    ERL_NIF_TERM* terms = (ERL_NIF_TERM*)enif_alloc(result_len * sizeof(ERL_NIF_TERM));
    for (unsigned int i = 0; i < result_len; i++) {
        terms[i] = enif_make_double(env, C[i]);
    }
    ERL_NIF_TERM result = enif_make_list_from_array(env, terms, result_len);

    enif_free(A);
    enif_free(B);
    enif_free(C);
    enif_free(terms);

    return result;
}

/* ============================================================================
 * SIMD SUM (REDUCTION)
 * ============================================================================ */

#if HAS_AVX
static double sum_avx(const double* a, size_t n) {
    __m256d sum = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        sum = _mm256_add_pd(sum, va);
    }
    /* Horizontal sum */
    __m128d low = _mm256_castpd256_pd128(sum);
    __m128d high = _mm256_extractf128_pd(sum, 1);
    __m128d sum128 = _mm_add_pd(low, high);
    sum128 = _mm_hadd_pd(sum128, sum128);
    double result;
    _mm_store_sd(&result, sum128);
    for (; i < n; i++) {
        result += a[i];
    }
    return result;
}
#endif

/**
 * NIF: simd_sum(List) -> Float
 */
static ERL_NIF_TERM simd_sum(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    unsigned int len;
    if (!enif_get_list_length(env, argv[0], &len) || len == 0) {
        return enif_make_badarg(env);
    }

    double* a = (double*)enif_alloc(len * sizeof(double));
    if (!a) return enif_make_badarg(env);

    ERL_NIF_TERM head, tail = argv[0];
    for (unsigned int i = 0; i < len; i++) {
        enif_get_list_cell(env, tail, &head, &tail);
        enif_get_double(env, head, &a[i]);
    }

    double result;
#if HAS_AVX
    result = sum_avx(a, len);
#else
    result = 0.0;
    for (unsigned int i = 0; i < len; i++) {
        result += a[i];
    }
#endif

    enif_free(a);
    return enif_make_double(env, result);
}

/* ============================================================================
 * SIMD SCALE (x * scalar)
 * ============================================================================ */

/**
 * NIF: simd_scale(List, Scalar) -> List
 */
static ERL_NIF_TERM simd_scale(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    unsigned int len;
    double scalar;

    if (!enif_get_list_length(env, argv[0], &len) ||
        !enif_get_double(env, argv[1], &scalar) ||
        len == 0) {
        return enif_make_badarg(env);
    }

    double* a = (double*)enif_alloc(len * sizeof(double));
    double* b = (double*)enif_alloc(len * sizeof(double));
    if (!a || !b) {
        if (a) enif_free(a);
        if (b) enif_free(b);
        return enif_make_badarg(env);
    }

    ERL_NIF_TERM head, tail = argv[0];
    for (unsigned int i = 0; i < len; i++) {
        enif_get_list_cell(env, tail, &head, &tail);
        enif_get_double(env, head, &a[i]);
    }

#if HAS_AVX
    __m256d vs = _mm256_set1_pd(scalar);
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vr = _mm256_mul_pd(va, vs);
        _mm256_storeu_pd(&b[i], vr);
    }
    for (; i < len; i++) {
        b[i] = a[i] * scalar;
    }
#else
    for (unsigned int i = 0; i < len; i++) {
        b[i] = a[i] * scalar;
    }
#endif

    ERL_NIF_TERM* terms = (ERL_NIF_TERM*)enif_alloc(len * sizeof(ERL_NIF_TERM));
    for (unsigned int i = 0; i < len; i++) {
        terms[i] = enif_make_double(env, b[i]);
    }
    ERL_NIF_TERM result = enif_make_list_from_array(env, terms, len);

    enif_free(a);
    enif_free(b);
    enif_free(terms);

    return result;
}

/* ============================================================================
 * NIF EXPORTS
 * ============================================================================ */

static ErlNifFunc nif_funcs[] = {
    {"simd_dot", 2, simd_dot, 0},
    {"simd_mul", 2, simd_mul, 0},
    {"simd_matmul", 5, simd_matmul, 0},
    {"simd_sum", 1, simd_sum, 0},
    {"simd_scale", 2, simd_scale, 0}
};

ERL_NIF_INIT(viva_simd_nif, nif_funcs, NULL, NULL, NULL, NULL)
