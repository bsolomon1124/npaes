# npaes

_in development..._

Advanced Encryption Standard (AES) NumPy implementation.

<table>
<tr>
  <td>Supports</td>
  <td>Python 2.7 | 3.5 | 3.6 | 3.7</td>
</tr>
</table>

This package implements the Advanced Encryption Standard (AES) as specified in Federal Information Processing Standards Publication 197 ("FIPS197"):

> [https://csrc.nist.gov/publications/detail/fips/197/final](https://csrc.nist.gov/publications/detail/fips/197/final)

This implementation is based entirely and solely on FIPS197.  The tests in `tests/test_npaes.py` use the full set of example vectors from Appendix A, B, and C of FIPS197.  `npaes` supports AES-128, AES-192, and AES-256.

Its sole dependencies is NumPy.  It does not use OpenSSL or any other C libraries besides the portions of NumPy that are implemented in C.

Additional technical notes can be found in the docstring of `npaes/__init__.py`.
