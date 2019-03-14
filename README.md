# npaes

Advanced Encryption Standard (AES) NumPy implementation.

<span style="color:red">**Warning**: incomplete & not yet suitable for production use.</span>  See the "Caution" section below.

----

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

## Caution

This package is incomplete.  While the raw encryption and decryption are fully tested using the FIPS197 example vectors, it is incomplete for the following reasons:

- It does not allow you to specify an [initialization vector](https://en.wikipedia.org/wiki/Initialization_vector) (IV).
- It does not allow you to specify a [block mode](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation).
- It is optimized in most places but not all, and has little to no chance of ever being as fast as the optimized ANSI C version in OpenSSL.

## Basic Usage

```python
import os
from npaes import AES

# Key must be bytes type and 128, 192, or 256 bits
# Or use hashlib.sha256() for an off-length key
key = os.urandom(32)  # 256-bit key

# Your plaintext length must be bytes and a multiple of 16 length
msg = b"a secret message goes here" + 6 * b"\x03"
cipher = AES(key)
ciphertext = cipher.encrypt(msg)

print(ciphertext)
# b'a\x85cna\xc2\xeeu\xe9S\xdf\xabE\x0c\xda\xf4\x19\x11\xa3!\xdd\x96-\x85\x10f\xd4\x18;s%\x81'
print(cipher.decrypt(ciphertext))
# b'a secret message goes here\x03\x03\x03\x03\x03\x03'
```
