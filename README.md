# npaes

[![PyPI version](https://img.shields.io/pypi/v/npaes.svg)](https://pypi.org/project/npaes/)
[![Python versions](https://img.shields.io/pypi/pyversions/npaes.svg)](https://pypi.org/project/npaes/)
[![License: Apache-2.0](https://img.shields.io/pypi/l/npaes.svg)](https://github.com/bsolomon1124/npaes/blob/master/LICENSE)

Advanced Encryption Standard (AES) NumPy implementation.

> **Warning:** incomplete and not suitable for production use.
> See the [Caution](#caution) section.

## Overview

This package implements the Advanced Encryption Standard (AES) as specified
in Federal Information Processing Standards Publication 197 ("FIPS 197"):

<https://csrc.nist.gov/publications/detail/fips/197/final>

It is based entirely and solely on FIPS 197. The tests in
`tests/test_npaes.py` use the full set of example vectors from Appendices A,
B, and C of FIPS 197. `npaes` supports AES-128, AES-192, and AES-256.

Its sole dependency is NumPy. It does not use OpenSSL or any other C
libraries besides the portions of NumPy that are implemented in C.

Additional technical notes can be found in the docstring of
`src/npaes/__init__.py`.

## Installation

From PyPI:

```bash
pip install npaes
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add npaes
```

## Basic usage

```python
import os

from npaes import AES

# Key must be bytes type and 128, 192, or 256 bits.
# Or use hashlib.sha256() for an off-length key.
key = os.urandom(32)  # 256-bit key

# Your plaintext must be bytes and a multiple of 16 bytes long.
msg = b"a secret message goes here" + 6 * b"\x03"
cipher = AES(key)
ciphertext = cipher.encrypt(msg)

print(ciphertext)
# b'a\x85cna\xc2\xeeu\xe9S\xdf\xabE\x0c\xda\xf4\x19\x11\xa3!\xdd\x96-\x85\x10f\xd4\x18;s%\x81'
print(cipher.decrypt(ciphertext))
# b'a secret message goes here\x03\x03\x03\x03\x03\x03'
```

## Caution

This package is incomplete. While the raw encryption and decryption are
fully tested using the FIPS 197 example vectors, it is incomplete for the
following reasons:

- It does not allow you to specify an
  [initialization vector](https://en.wikipedia.org/wiki/Initialization_vector)
  (IV).
- It does not allow you to specify a
  [block mode](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation).
- It is optimized in most places but not all, and has little to no chance of
  ever being as fast as the optimized ANSI C version in OpenSSL.

## Development

This project uses [uv](https://docs.astral.sh/uv/) with the `uv_build`
backend and a `src/` layout.

```bash
uv sync                                         # install deps
uv run pytest                                   # tests + coverage
uv run ruff check && uv run ruff format --check # lint + format
uv run ty check                                 # type check
uv build                                        # build sdist + wheel
```

See [`CHANGELOG.md`](CHANGELOG.md) for release notes.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
