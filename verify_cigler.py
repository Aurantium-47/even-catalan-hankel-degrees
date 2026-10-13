#!/usr/bin/env python3
"""Exact checks for the uniform Catalan-convolution residue-degree proof.

Python 3.10+; standard library only. No network, floating point, or CAS is used.
Run:
    python verify_cigler.py
To regenerate the two deterministic data files from scratch:
    python verify_cigler.py --write-data

The infinite-parameter proof is in main.tex. These
computations are independent checks, not an inference from a finite sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction as F
from math import comb, factorial
from pathlib import Path
from typing import TypeAlias

Laurent: TypeAlias = dict[int, F]


def sign(e: int) -> int:
    return -1 if e & 1 else 1


def choose(a: int, b: int) -> int:
    """The falling-factorial binomial coefficient at any integer a."""
    if b < 0:
        raise ValueError('b must be nonnegative')
    if a >= 0:
        return comb(a, b) if a >= b else 0
    return sign(b) * comb(b-a-1, b)


def det_integer(matrix: list[list[int]]) -> int:
    """Fraction-free Bareiss elimination with exact divisions and pivoting."""
    a = [row[:] for row in matrix]
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError('matrix must be square')
    if n == 0:
        return 1
    previous, orientation = 1, 1
    for p in range(n-1):
        if a[p][p] == 0:
            r = next((r for r in range(p+1, n) if a[r][p]), None)
            if r is None:
                return 0
            a[p], a[r] = a[r], a[p]
            orientation = -orientation
        pivot = a[p][p]
        for i in range(p+1, n):
            for j in range(p+1, n):
                numerator = pivot*a[i][j] - a[i][p]*a[p][j]
                a[i][j], remainder = divmod(numerator, previous)
                if remainder:
                    raise ArithmeticError('non-exact Bareiss division')
            a[i][p] = 0
        previous = pivot
    return orientation*a[-1][-1]


def det_rational(matrix: list[list[F]]) -> F:
    """Ordinary Gaussian elimination, separate from det_integer."""
    a = [[F(x) for x in row] for row in matrix]
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError('matrix must be square')
    answer = F(1)
    for p in range(n):
        r = next((r for r in range(p, n) if a[r][p]), None)
        if r is None:
            return F(0)
        if r != p:
            a[p], a[r] = a[r], a[p]
            answer = -answer
        pivot = a[p][p]
        answer *= pivot
        for i in range(p+1, n):
            factor = a[i][p]/pivot
            for j in range(p+1, n):
                a[i][j] -= factor*a[p][j]
    return answer


def moment(k: int, m: int) -> int:
    if k < 1 or m < 0:
        raise ValueError('k >= 1 and m >= 0 required')
    value, remainder = divmod(2*k*comb(2*m+2*k, m), 2*m+2*k)
    if remainder:
        raise ArithmeticError('Catalan-convolution coefficient not integral')
    return value


def direct_hankel(k: int, size: int) -> int:
    if size < 0:
        raise ValueError('a Hankel size cannot be negative')
    a = [moment(k, m) for m in range(max(0, 2*size-1))]
    return det_integer([[a[i+j] for j in range(size)] for i in range(size)])


def chebyshev_recurrence(m: int) -> list[int]:
    """P_m(x), with P_0=1, P_1=x-2; coefficients low degree first."""
    if m < 0:
        raise ValueError('nonnegative index required here')
    a, b = [1], [-2, 1]
    if m == 0:
        return a
    for _ in range(1, m):
        c = [0] + b
        for i, v in enumerate(b):
            c[i] -= 2*v
        for i, v in enumerate(a):
            c[i] -= v
        a, b = b, c
    return b


def remainders(k: int, max_index: int) -> list[list[int]]:
    """P_m mod x^(k-1) P_(k-1), constructed only by the recurrence."""
    if k < 2:
        raise ValueError('k >= 2 required')
    s, width = k-1, 2*k-2
    q = [0]*s + chebyshev_recurrence(s)
    rows = [[1]+[0]*(width-1), [-2, 1]+[0]*(width-2)]
    for _ in range(1, max_index):
        a, b = rows[-2], rows[-1]
        c = [0]+b
        for i in range(width):
            c[i] -= 2*b[i]+a[i]
        lead = c.pop()
        for i in range(width):
            c[i] -= lead*q[i]
        rows.append(c)
    return rows


def extended_R(k: int, size: int, table: list[list[int]] | None = None) -> int:
    """Algebraic remainder determinant, not a negative-size Hankel determinant."""
    width = 2*k-2
    indices = list(range(size, size+width))
    largest = max(0, *(m if m >= 0 else -m-2 for m in indices))
    if table is None:
        table = remainders(k, largest)
    columns = []
    for m in indices:
        if m == -1:
            columns.append([0]*width)
        elif m >= 0:
            columns.append(table[m])
        else:
            columns.append([-v for v in table[-m-2]])
    return det_integer([[col[i] for col in columns] for i in range(width)])


def quotient_coefficients(k: int, n: int, order: int) -> list[F]:
    """[x^r] P_(M+k-1)/P_(M-1) for M=k(n+1), using binomial coefficients.

    The common parity (-1)^(M-1) is removed before series division.
    This works also for negative n except n=-1.
    """
    M = k*(n+1)
    if M == 0:
        raise ValueError('the raw quotient is singular at n=-1; use its polynomial')
    p = [sign(i)*choose(M+i, 2*i+1) for i in range(order+1)]
    q = [sign(k+i)*choose(M+k+i, 2*i+1) for i in range(order+1)]
    h: list[F] = []
    for i in range(order+1):
        h.append((F(q[i])-sum(p[b]*h[i-b] for b in range(1, i+1)))/M)
    return h


def quotient_value(k: int, j: int, n: int) -> F:
    if not (1 <= j and k >= max(2, 2*j-1)):
        raise ValueError('outside the theorem range')
    s, d, t = k-1, j-1, k-j
    h = quotient_coefficients(k, n, s-1)
    minor = det_rational([[h[t+a-b] for b in range(d)] for a in range(d)])
    return sign(comb(j, 2))*F(n+1)**s*minor


def clean(a: Laurent) -> Laurent:
    return {i: F(v) for i, v in a.items() if v}


def add(a: Laurent, b: Laurent, scale_b: F = F(1)) -> Laurent:
    c = a.copy()
    for i, v in b.items():
        c[i] = c.get(i, F(0)) + scale_b*v
    return clean(c)


def mul(a: Laurent, b: Laurent) -> Laurent:
    c: Laurent = {}
    for i, u in a.items():
        for j, v in b.items():
            c[i+j] = c.get(i+j, F(0)) + u*v
    return clean(c)


def scale_shift(a: Laurent, coefficient: F, shift: int = 0) -> Laurent:
    return clean({i+shift: coefficient*v for i, v in a.items()})


def binomial_polynomial(offset: int, order: int) -> Laurent:
    """binom(M+offset,order) exactly in Q[M]."""
    result = {0: F(1)}
    for i in range(order):
        result = mul(result, {1: F(1), 0: F(offset-i)})
    return scale_shift(result, F(1, factorial(order)))


def symbolic_quotient(k: int, order: int) -> list[Laurent]:
    p = [scale_shift(binomial_polynomial(i, 2*i+1), F(sign(i)))
         for i in range(order+1)]
    q = [scale_shift(binomial_polynomial(k+i, 2*i+1), F(sign(k+i)))
         for i in range(order+1)]
    h: list[Laurent] = []
    for i in range(order+1):
        numerator = q[i]
        for b in range(1, i+1):
            numerator = add(numerator, mul(p[b], h[i-b]), F(-1))
        h.append(scale_shift(numerator, F(1), -1))
        if h[-1]:
            assert min(h[-1]) >= -1
            assert max(h[-1]) <= max(0, 2*i-1)
    return h


def symbolic_j3(k: int) -> list[F]:
    """Exact coefficients a_e of f_(k,3)(n)=sum a_e (n+1)^e."""
    if k < 5:
        raise ValueError('k >= 5 required')
    s, t = k-1, k-3
    h = symbolic_quotient(k, s-1)
    minor = add(mul(h[t], h[t]), mul(h[t-1], h[t+1]), F(-1))
    # f = -(M/k)^s * minor, then M=k(n+1).
    f_in_M = scale_shift(minor, F(-1, k**s), s)
    assert f_in_M and min(f_in_M) >= 0
    result = [F(0)]*(max(f_in_M)+1)
    for e, coefficient in f_in_M.items():
        result[e] = coefficient*k**e
    return result


def evaluate_shifted(coefficients: list[F], n: int) -> F:
    value = F(0)
    for a in reversed(coefficients):
        value = value*(n+1)+a
    return value


def bernoulli_numbers(max_index: int) -> list[F]:
    b = [F(1)]
    for m in range(1, max_index+1):
        b.append(-sum(F(comb(m+1, i))*b[i] for i in range(m))/(m+1))
    return b


def beta_values(max_index: int) -> list[F]:
    b = bernoulli_numbers(2*max_index)
    return [F(0)] + [F(sign(m+1)*2**(2*m))*b[2*m]/factorial(2*m)
                      for m in range(1, max_index+1)]


def leading_j3(k: int, beta: list[F]) -> F:
    t = k-3
    return k**(4*t)*(beta[t-1]*beta[t+1]-beta[t]**2)


def newton(values: list[int]) -> list[int]:
    a = values[:]
    out = []
    while a:
        out.append(a[0])
        a = [v-u for u, v in zip(a, a[1:])]
    return out


def run_suite(directory: Path) -> tuple[dict, dict]:
    beta = beta_values(50)
    records: dict[str, object] = {}
    polynomials: dict[str, object] = {
        'basis': 'f_(k,3)(n) = sum_e coefficients[e] * (n+1)^e',
        'coefficient_encoding': 'exact rational strings; omitted none, zero is 0',
        'reflected_residue': 'f_(k,k-2)(n) = (-1)^(k-1+binom(k,2)) f_(k,3)(-n-3)',
        'polynomials': []
    }
    chosen_k = list(range(5, 25)) + [32, 40, 50]
    symbolic: dict[int, list[F]] = {}
    for k in chosen_k:
        a = symbolic_j3(k)
        expected = leading_j3(k, beta)
        assert len(a)-1 == 5*(k-3)
        assert a[-1] == expected and expected > 0
        symbolic[k] = a
        polynomials['polynomials'].append({
            'k': k, 'degree': len(a)-1,
            'leading_coefficient': str(expected),
            'coefficients': [str(c) for c in a]
        })
    records['symbolic_k'] = chosen_k
    records['symbolic_polynomials'] = len(chosen_k)
    # Exact boundary factorization in y=n+1, using polynomial multiplication.
    factored = {2: F(1)}
    for factor in ({0: F(1), 1: F(2), 2: F(1)},
                   {0: F(1), 1: F(4), 2: F(4)},
                   {0: F(-342), 1: F(-1375), 2: F(-875),
                    3: F(1000), 4: F(500)}):
        factored = mul(factored, factor)
    factored = scale_shift(factored, F(125, 3024))
    assert [factored.get(e, F(0)) for e in range(11)] == symbolic[5]
    records['k5_factorization_verified'] = True

    # Original N-by-N determinants use only the explicit Catalan coefficients.
    direct = []
    for k in range(5, 13):
        for n in range(4):
            for r in sorted({3, k-2}):
                actual = sign(n*comb(k, 2))*direct_hankel(k, k*n+r)
                expected = (evaluate_shifted(symbolic[k], n) if r == 3 else
                    sign(k-1+comb(k, 2))*evaluate_shifted(symbolic[k], -n-3))
                assert actual == expected
                direct.append({'k': k, 'r': r, 'n': n, 'value': str(actual)})
    assert direct_hankel(5, 0) == 1
    records['direct_hankel_checks'] = direct

    # Signed extension and reflection are checked at negative n as well,
    # including n=-1 where the quotient representation itself is singular.
    reduced = []
    for k in range(5, 25):
        n_values = list(range(-4, 7))
        largest = max(k*6+k-2+2*k-3, 4*k+2*k)
        table = remainders(k, largest)
        for n in n_values:
            for r in sorted({3, k-2}):
                actual = sign(n*comb(k, 2))*extended_R(k, k*n+r, table)
                expected = (evaluate_shifted(symbolic[k], n) if r == 3 else
                    sign(k-1+comb(k, 2))*evaluate_shifted(symbolic[k], -n-3))
                assert actual == expected, (k, n, r)
                reduced.append({'k': k, 'r': r, 'n': n, 'value': str(actual)})
    records['remainder_checks'] = reduced

    # General-j quotient identity. This is a check of the stronger theorem,
    # including the extremal case k=2j-1, not used to extrapolate a theorem.
    general = []
    for k in range(2, 19):
        table = remainders(k, 5*k)
        for j in range(1, (k+1)//2+1):
            d, t = j-1, k-j
            if d:
                first = k-2*j+2
                gram = det_rational([[beta[first+a+b] for b in range(d)]
                                     for a in range(d)])
                assert gram > 0
            for n in range(3):
                actual = sign(n*comb(k, 2))*extended_R(k, k*n+j, table)
                expected = quotient_value(k, j, n)
                assert actual == expected, (k, j, n)
                general.append({'k': k, 'j': j, 'n': n, 'value': str(actual)})
    records['general_j_checks'] = general

    cert_path = directory/'k16_certificate.json'
    certificate = json.loads(cert_path.read_text())
    assert certificate['k'] == 16
    copied_hash = hashlib.sha256(cert_path.read_bytes()).hexdigest()
    assert copied_hash == '711d648db738c091dee34f344eb8aa56030500a1ce7c4ae7b40a2c551fa3213d'
    tested = []
    for r_text, saved in certificate['values'].items():
        r = int(r_text)
        values = []
        for n in range(226):
            scalar = (quotient_value(16, 3, n) if r == 3 else
                sign(15+comb(16, 2))*quotient_value(16, 3, -n-3))
            symbolic_value = (evaluate_shifted(symbolic[16], n) if r == 3 else
                sign(15+comb(16, 2))*evaluate_shifted(symbolic[16], -n-3))
            assert scalar == symbolic_value and scalar.denominator == 1
            values.append(scalar.numerator)
        assert values == saved['values']
        differences = newton(values)
        assert differences[:66] == saved['newton_coefficients']
        assert all(v == 0 for v in differences[66:])
        leading = F(differences[65], factorial(65))
        assert leading == leading_j3(16, beta)
        assert leading == F(saved['leading_num'], saved['leading_den'])
        tested.append({'r': r, 'values_verified': 226, 'degree': 65,
                       'leading_coefficient': str(leading)})
    records['k16_certificate'] = {
        'sha256': copied_hash, 'residues': tested,
        'note': 'Compared to both the symbolic polynomial and the scalar quotient.'
    }
    records['counts'] = {
        'symbolic_j3_polynomials': len(chosen_k),
        'direct_hankel': len(direct),
        'remainder_determinants': len(reduced),
        'general_j_quotient': len(general),
        'k16_values': sum(r['values_verified'] for r in tested),
        'empty_hankel': 1
    }
    records['all_checks_passed'] = True
    return records, polynomials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--write-data', action='store_true',
                        help='write exact_checks.json and exact_polynomials.json')
    options = parser.parse_args()
    records, polynomials = run_suite(options.data_dir)
    for name, data in [('exact_checks.json', records), ('exact_polynomials.json', polynomials)]:
        target = options.data_dir/name
        if options.write_data:
            target.write_text(json.dumps(data, indent=2)+'\n')
        else:
            if not target.is_file():
                raise FileNotFoundError(f'{target}: run once with --write-data')
            assert json.loads(target.read_text()) == data, f'data mismatch: {name}'
    print(json.dumps({'all_checks_passed': True, **records['counts']}, indent=2))


if __name__ == '__main__':
    main()
