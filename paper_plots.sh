# ===========================================
#         LINEAR ON COMPACT DOMAINS
# ===========================================

# === Sublinear convergence ===
# ... for uniform and optimal sampling
for Z in uniform optimal; do
    python linear_sgd.py \
        -b legendre \
        -d 3 \
        -T exp \
        -t 10 \
        -s deterministic_unbounded \
        -Z ${Z} \
        -z 1 \
        -S inf \
        -I random \
        -i 100000 \
        -p quasi
done

# === Exponential convergence ===
# ... for uniform and optimal sampling
for Z in uniform optimal; do
    python linear_sgd.py \
        -b legendre \
        -d 3 \
        -T exp \
        -t 10 \
        -s constant \
        -Z ${Z} \
        -z 1 \
        -S inf \
        -I random \
        -i 10000 \
        -p quasi
done

# === Exponential convergence with mixed step size rule ===
# ... for uniform and optimal sampling
for Z in uniform optimal; do
    python linear_sgd.py \
        -b legendre \
        -d 3 \
        -T exp \
        -t 10 \
        -s mixed \
        -Z ${Z} \
        -z 1 \
        -S inf \
        -I random \
        -i 100000 \
        -p quasi
done

# === Exponential convergence to κ for biased estimators ===
# ... for stabilised quasi-projection and least squares projection
for p in quasi least-squares; do
    python linear_sgd.py \
        -b legendre \
        -d 3 \
        -T exp \
        -t 10 \
        -s constant \
        -Z optimal \
        -z 9 \
        -S 0.5 \
        -I random \
        -i 100 \
        -p ${p}
done

# === Exponential convergence up to κ for biased estimators (s=1) ===
# ... for multiple stability parameters ...
for S in 0.5 0.25 0.125 0.0625; do
    # ... and stabilised quasi-projection and least squares projection
    for p in quasi least-squares; do
        python linear_sgd.py \
            -b legendre \
            -d 3 \
            -T exp \
            -t 10 \
            -s 1 \
            -Z optimal \
            -z 9 \
            -S ${S} \
            -I random \
            -i 10 \
            -p ${p}
    done
done


# ===========================================
#        LINEAR ON UNBOUNDED DOMAINS
# ===========================================

# === (Non-)convergence for Hermite polynomials ===
python linear_sgd_hermite.py \
    -d 7 \
    -T exp \
    -t 10 \
    -s deterministic_unbounded \
    -Z optimal \
    -z 7 \
    -S inf \
    -I random \
    -i 10000 \
    -p quasi

for z in 1 21 70; do
    python linear_sgd_hermite.py \
        -d 7 \
        -T exp \
        -t 10 \
        -s deterministic_unbounded \
        -Z gaussian \
        -z ${z} \
        -S inf \
        -I random \
        -i 10000 \
        -p quasi
done

# ===========================================
#     SHALLOW NETWORKS ON COMPACT DOMAINS
# ===========================================
