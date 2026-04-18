"""Advanced Encryption Standard (AES) NumPy implementation.

Based strictly on:

    Federal Information Processing Standards Publication 197
    https://csrc.nist.gov/publications/detail/fips/197/final

Any reference to the paper in this source code is called just FIPS197.

See also the original submission:

    The Rijndael Block Cipher
    https://csrc.nist.gov/archive/aes/rijndael/Rijndael-ammended.pdf
---------------------------------------------

Rijndael Algorithm: Short Description & Technical Notes:

 - Symmetric block cipher that can process data blocks of 128 bits, using
   cipher keys with lengths of 128, 192, and 256 bits.  Here, the data
   block is a 4x4 NumPy array, with each cell representing 8 bits:

                  Nk     Nb    Nr
       AES-128     4      4    10
       AES-192     6      4    12
       AAS-256     8      4    14

 - Internally, the AES algorithm's operations are performed on a
   two-dimensional array of bytes called the State. The State consists
   of four rows of bytes, each containing Nb bytes, where Nb is the
   block length divided by 32.

 - A *word* is "a group of 32 bits that is treated either as a
   single entity or as an array of 4 bytes."

 - NumPy arrays are by default row-ordered ("C-style") rather than
   column-ordered, ("Fortran-style").  FIPS197's way of thinking is
   better described as Fortran-style because it depicts the state as an
   "array of columns," where each column represents a word.  That is,
   given the 16-byte input:

       {in0, in1, in2, in3, in4, in5, ..., in13, in14, in15}

   FIPS197 will reshape as:

       [[in0, in4, in8,  in12],
        [in1, in5, in9,  in13],
        [in2, in6, in10, in14],
        [in3, in7, in11, in15]], with word0 as {in0, in1, in2, in3}

   whereas NumPy C-style row-ordering would call for:

       [[in0,  in1,  in2,  in3],
        [in4,  in5,  in6,  in7],
        [in8,  in9,  in10, in11],
        [in12, in13, in14, in15]], with word0 as {in0, in1, in2, in3}

   i.e. "The state can hence be interpreted as a one-dimensional
   array of 32 bit words (columns), w0...w3, where the column number
   c provides an index into this array." (section 3.5)

   Note lastly that specifying order="F" doesn't really do much
   for our purposes here; for instance, `.reshape()` will still behave
   the same.  We use `.swapaxes(0, 1)` to follow FIPS197 from the getgo
   from a given input.

 - The XOR (addition or "⊕" in FIPS paper) is the bitwise exclusive-or.
   Example: The number 13 is represented by 00001101. Likewise, 17 is
   represented by 00010001. The bit-wise XOR of 13 and 17 is
   therefore 00011100, or 28.
   This is NumPy's `np.bitwise_xor()`, *not* `np.logical_xor()`.
   See `np.bitwise_xor(0b00001101, 0b00010001)` (binary literal) or
   `np.bitwise_xor(0x0d, 0x11)` (hex literal).

 - FIPS197 frequently uses the practice, which is common in C, of
   passing the State array as a function input, and then modifying
   that array inplace in the body of the function.  That is, the
   original 16-byte input array is copied into an empty 4x4 state
   array, and after that (almost) no copies are made; State is
   manipulated inplace rather than assignment to a new object.  The
   only other copy that is made is that State is copied into an Out
   array which gets returned in the main function:

       in -> State -> out

   We (currently) do things less efficiently, using copies in most
   places, though may be able to get rid of this using the `out`
   argument to NumPy ufuncs (Consider this a TODO.)  If we could do
   this everywhere, the advantage would be in avoiding the allocation
   of new memory where it is not necessary.

   Note that the result dtype must match the original input type
   if you write a result into the original array.  This will work:

   >>> a = array([2, 4, 6, 8], dtype=np.uint8)
   >>> np.add(a, 2, out=a)
   array([ 4,  6,  8, 10], dtype=uint8)
   >>> a
   array([ 4,  6,  8, 10], dtype=uint8)

   However, this will raise because it would entail a change in dtype
   from uint8 to float
   >>> np.divide(a, 2, out=a)  # raises TypeError

 - We use vectorization *where possible*, but because of the nonlinear
   nature of much of AES, and that computation of each round depends
   on result of the last, Python loops do sneak into some places.

 - One last technical note is that, because we're working with bits,
   we work almost exclusively with np.uint8 dtype.
"""

from __future__ import annotations

__all__ = ("AES",)
__version__ = "0.4"

import functools
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy import arange, array, int16, uint8
from numpy import bitwise_xor as xor
from numpy.lib.stride_tricks import as_strided
from numpy.typing import NDArray

# PEP 695 `type` statement would be cleaner but requires Python 3.12+.
# This project supports 3.10+, so we use TypeAlias instead.
KeyLengthBytes: TypeAlias = Literal[16, 24, 32]
Nk: TypeAlias = Literal[4, 6, 8]
Nr: TypeAlias = Literal[10, 12, 14]
UInt8Array: TypeAlias = NDArray[np.uint8]


class AES:
    key: bytes

    def __init__(self, key: bytes) -> None:  # TODO: iv
        if not isinstance(key, bytes):
            raise TypeError(f"`key` must be bytes, not {type(key)}")
        if len(key) not in ALLOWED_KEYLENGTH_BYTES:
            raise ValueError(f"len(key) must be 16, 24, or 32 bytes, not {len(key)}")
        self.key = key

    def encrypt(self, plaintext: bytes) -> bytes:
        if not isinstance(plaintext, bytes):
            raise TypeError(f"`plaintext` must be bytes, not {type(plaintext)}")
        if len(plaintext) % BLOCKSIZE_BYTES != 0:
            raise ValueError(
                "len(plaintext) should be a multiple of 16"
                " (AES encrypts and decrypts in 128-bit blocks)."
                " Pad the input first."
            )
        parray = plaintext_to_3darray(plaintext)  # 3d
        karray = key_to_array(self.key)
        # TODO: we could probably vectorize this and just operate with everything
        # in 3 dimensions rather than 2
        ciphertext = b""
        for p in parray:
            ciphertext += array_to_bytes(encrypt_raw(p, karray))
        return ciphertext

    def decrypt(self, ciphertext: bytes) -> bytes:
        if not isinstance(ciphertext, bytes):
            raise TypeError(f"`ciphertext` must be bytes, not {type(ciphertext)}")
        carray = plaintext_to_3darray(ciphertext)  # 3d
        karray = key_to_array(self.key)
        # TODO: we could probably vectorize this and just operate with everything
        # in 3 dimensions rather than 2
        plaintext = b""
        for c in carray:
            plaintext += array_to_bytes(decrypt_raw(c, karray))
        return plaintext


# Rijndael processes data blocks of 128 bits
BLOCKSIZE_BITS = 128
BLOCKSIZE_BYTES = 16

# Rinjdael allows Cipher Keys with lengths of 128, 192, or 256 bits,
# corresponding to 16, 24, and 32 bytes respectively
ALLOWED_KEYLENGTH_BITS = frozenset({128, 192, 256})
ALLOWED_KEYLENGTH_BYTES = frozenset({16, 24, 32})

# Number of columns (32-bit words) comprising the State.
# "Future reaffirmations of this standard could include changes" but,
# "for this standard, Nb = 4."
NB = 4

# Number of 32-bit words comprising the Cipher Key
ALLOWED_NK = frozenset({4, 6, 8})


def numrounds(nk: Nk, _table={4: 10, 6: 12, 8: 14}) -> Nr:
    """Return number of rounds (Nr) as function of key size (Nk).

    It is technically a function of Nk and Nb, but Nb is fixed at 4.
    """
    return cast(Nr, _table[nk])


# ---------------------------------------------------------------------
# Core encryption functions:
# SubBytes(), ShiftRows(), MixColumns()
# (AddRoundKey() is just an XOR)

SBOX = array(
    [
        #  0     1     2     3     4     5     6     7     8     9     a     b     c     d     e     f
        0x63,
        0x7C,
        0x77,
        0x7B,
        0xF2,
        0x6B,
        0x6F,
        0xC5,
        0x30,
        0x01,
        0x67,
        0x2B,
        0xFE,
        0xD7,
        0xAB,
        0x76,  # 0
        0xCA,
        0x82,
        0xC9,
        0x7D,
        0xFA,
        0x59,
        0x47,
        0xF0,
        0xAD,
        0xD4,
        0xA2,
        0xAF,
        0x9C,
        0xA4,
        0x72,
        0xC0,  # 1
        0xB7,
        0xFD,
        0x93,
        0x26,
        0x36,
        0x3F,
        0xF7,
        0xCC,
        0x34,
        0xA5,
        0xE5,
        0xF1,
        0x71,
        0xD8,
        0x31,
        0x15,  # 2
        0x04,
        0xC7,
        0x23,
        0xC3,
        0x18,
        0x96,
        0x05,
        0x9A,
        0x07,
        0x12,
        0x80,
        0xE2,
        0xEB,
        0x27,
        0xB2,
        0x75,  # 3
        0x09,
        0x83,
        0x2C,
        0x1A,
        0x1B,
        0x6E,
        0x5A,
        0xA0,
        0x52,
        0x3B,
        0xD6,
        0xB3,
        0x29,
        0xE3,
        0x2F,
        0x84,  # 4
        0x53,
        0xD1,
        0x00,
        0xED,
        0x20,
        0xFC,
        0xB1,
        0x5B,
        0x6A,
        0xCB,
        0xBE,
        0x39,
        0x4A,
        0x4C,
        0x58,
        0xCF,  # 5
        0xD0,
        0xEF,
        0xAA,
        0xFB,
        0x43,
        0x4D,
        0x33,
        0x85,
        0x45,
        0xF9,
        0x02,
        0x7F,
        0x50,
        0x3C,
        0x9F,
        0xA8,  # 6
        0x51,
        0xA3,
        0x40,
        0x8F,
        0x92,
        0x9D,
        0x38,
        0xF5,
        0xBC,
        0xB6,
        0xDA,
        0x21,
        0x10,
        0xFF,
        0xF3,
        0xD2,  # 7
        0xCD,
        0x0C,
        0x13,
        0xEC,
        0x5F,
        0x97,
        0x44,
        0x17,
        0xC4,
        0xA7,
        0x7E,
        0x3D,
        0x64,
        0x5D,
        0x19,
        0x73,  # 8
        0x60,
        0x81,
        0x4F,
        0xDC,
        0x22,
        0x2A,
        0x90,
        0x88,
        0x46,
        0xEE,
        0xB8,
        0x14,
        0xDE,
        0x5E,
        0x0B,
        0xDB,  # 9
        0xE0,
        0x32,
        0x3A,
        0x0A,
        0x49,
        0x06,
        0x24,
        0x5C,
        0xC2,
        0xD3,
        0xAC,
        0x62,
        0x91,
        0x95,
        0xE4,
        0x79,  # a
        0xE7,
        0xC8,
        0x37,
        0x6D,
        0x8D,
        0xD5,
        0x4E,
        0xA9,
        0x6C,
        0x56,
        0xF4,
        0xEA,
        0x65,
        0x7A,
        0xAE,
        0x08,  # b
        0xBA,
        0x78,
        0x25,
        0x2E,
        0x1C,
        0xA6,
        0xB4,
        0xC6,
        0xE8,
        0xDD,
        0x74,
        0x1F,
        0x4B,
        0xBD,
        0x8B,
        0x8A,  # c
        0x70,
        0x3E,
        0xB5,
        0x66,
        0x48,
        0x03,
        0xF6,
        0x0E,
        0x61,
        0x35,
        0x57,
        0xB9,
        0x86,
        0xC1,
        0x1D,
        0x9E,  # d
        0xE1,
        0xF8,
        0x98,
        0x11,
        0x69,
        0xD9,
        0x8E,
        0x94,
        0x9B,
        0x1E,
        0x87,
        0xE9,
        0xCE,
        0x55,
        0x28,
        0xDF,  # e
        0x8C,
        0xA1,
        0x89,
        0x0D,
        0xBF,
        0xE6,
        0x42,
        0x68,
        0x41,
        0x99,
        0x2D,
        0x0F,
        0xB0,
        0x54,
        0xBB,
        0x16,  # f
    ],
    dtype=uint8,
)


def sub_bytes(state: UInt8Array, out: UInt8Array | None = None, _sbox=SBOX) -> UInt8Array:
    if out is not None:
        out[:] = _sbox[state]
        return out
    return _sbox[state]


# Cols is also
# np.atleast_2d(arange(4)) - array([0, 3, 2, 1])[:, None]
# ...but that doesn't seem any simpler now, does it?
colindexer = array([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]], dtype=uint8)


def shift_rows(
    state: UInt8Array,
    out: UInt8Array | None = None,
    _rows=arange(4, dtype=uint8)[:, None],
    _cols=colindexer,
) -> UInt8Array:
    """Cyclically shift last 3 rows in the State."""
    if out is not None:
        out[:] = state[_rows, _cols]
        return out
    return state[_rows, _cols]


# ETABLE and LTABLE are lookup tables for matrix multiplication in GF(2^8)
ETABLE = array(
    [
        #  0     1     2     3     4     5     6     7     8     9     a     b     c     d     e     f
        0x01,
        0x03,
        0x05,
        0x0F,
        0x11,
        0x33,
        0x55,
        0xFF,
        0x1A,
        0x2E,
        0x72,
        0x96,
        0xA1,
        0xF8,
        0x13,
        0x35,  # 0
        0x5F,
        0xE1,
        0x38,
        0x48,
        0xD8,
        0x73,
        0x95,
        0xA4,
        0xF7,
        0x02,
        0x06,
        0x0A,
        0x1E,
        0x22,
        0x66,
        0xAA,  # 1
        0xE5,
        0x34,
        0x5C,
        0xE4,
        0x37,
        0x59,
        0xEB,
        0x26,
        0x6A,
        0xBE,
        0xD9,
        0x70,
        0x90,
        0xAB,
        0xE6,
        0x31,  # 2
        0x53,
        0xF5,
        0x04,
        0x0C,
        0x14,
        0x3C,
        0x44,
        0xCC,
        0x4F,
        0xD1,
        0x68,
        0xB8,
        0xD3,
        0x6E,
        0xB2,
        0xCD,  # 3
        0x4C,
        0xD4,
        0x67,
        0xA9,
        0xE0,
        0x3B,
        0x4D,
        0xD7,
        0x62,
        0xA6,
        0xF1,
        0x08,
        0x18,
        0x28,
        0x78,
        0x88,  # 4
        0x83,
        0x9E,
        0xB9,
        0xD0,
        0x6B,
        0xBD,
        0xDC,
        0x7F,
        0x81,
        0x98,
        0xB3,
        0xCE,
        0x49,
        0xDB,
        0x76,
        0x9A,  # 5
        0xB5,
        0xC4,
        0x57,
        0xF9,
        0x10,
        0x30,
        0x50,
        0xF0,
        0x0B,
        0x1D,
        0x27,
        0x69,
        0xBB,
        0xD6,
        0x61,
        0xA3,  # 6
        0xFE,
        0x19,
        0x2B,
        0x7D,
        0x87,
        0x92,
        0xAD,
        0xEC,
        0x2F,
        0x71,
        0x93,
        0xAE,
        0xE9,
        0x20,
        0x60,
        0xA0,  # 7
        0xFB,
        0x16,
        0x3A,
        0x4E,
        0xD2,
        0x6D,
        0xB7,
        0xC2,
        0x5D,
        0xE7,
        0x32,
        0x56,
        0xFA,
        0x15,
        0x3F,
        0x41,  # 8
        0xC3,
        0x5E,
        0xE2,
        0x3D,
        0x47,
        0xC9,
        0x40,
        0xC0,
        0x5B,
        0xED,
        0x2C,
        0x74,
        0x9C,
        0xBF,
        0xDA,
        0x75,  # 9
        0x9F,
        0xBA,
        0xD5,
        0x64,
        0xAC,
        0xEF,
        0x2A,
        0x7E,
        0x82,
        0x9D,
        0xBC,
        0xDF,
        0x7A,
        0x8E,
        0x89,
        0x80,  # a
        0x9B,
        0xB6,
        0xC1,
        0x58,
        0xE8,
        0x23,
        0x65,
        0xAF,
        0xEA,
        0x25,
        0x6F,
        0xB1,
        0xC8,
        0x43,
        0xC5,
        0x54,  # b
        0xFC,
        0x1F,
        0x21,
        0x63,
        0xA5,
        0xF4,
        0x07,
        0x09,
        0x1B,
        0x2D,
        0x77,
        0x99,
        0xB0,
        0xCB,
        0x46,
        0xCA,  # c
        0x45,
        0xCF,
        0x4A,
        0xDE,
        0x79,
        0x8B,
        0x86,
        0x91,
        0xA8,
        0xE3,
        0x3E,
        0x42,
        0xC6,
        0x51,
        0xF3,
        0x0E,  # d
        0x12,
        0x36,
        0x5A,
        0xEE,
        0x29,
        0x7B,
        0x8D,
        0x8C,
        0x8F,
        0x8A,
        0x85,
        0x94,
        0xA7,
        0xF2,
        0x0D,
        0x17,  # e
        0x39,
        0x4B,
        0xDD,
        0x7C,
        0x84,
        0x97,
        0xA2,
        0xFD,
        0x1C,
        0x24,
        0x6C,
        0xB4,
        0xC7,
        0x52,
        0xF6,
        0x01,  # f
    ],
    dtype=uint8,
)

# There is no value at 0,0 here, just a placeholder.  dtype is int16, not
# uint8, because we will be adding two of these together
LTABLE = array(
    [
        #  0     1     2     3     4     5     6     7     8     9     a     b     c     d     e     f
        0x00,
        0x00,
        0x19,
        0x01,
        0x32,
        0x02,
        0x1A,
        0xC6,
        0x4B,
        0xC7,
        0x1B,
        0x68,
        0x33,
        0xEE,
        0xDF,
        0x03,  # 0
        0x64,
        0x04,
        0xE0,
        0x0E,
        0x34,
        0x8D,
        0x81,
        0xEF,
        0x4C,
        0x71,
        0x08,
        0xC8,
        0xF8,
        0x69,
        0x1C,
        0xC1,  # 1
        0x7D,
        0xC2,
        0x1D,
        0xB5,
        0xF9,
        0xB9,
        0x27,
        0x6A,
        0x4D,
        0xE4,
        0xA6,
        0x72,
        0x9A,
        0xC9,
        0x09,
        0x78,  # 2
        0x65,
        0x2F,
        0x8A,
        0x05,
        0x21,
        0x0F,
        0xE1,
        0x24,
        0x12,
        0xF0,
        0x82,
        0x45,
        0x35,
        0x93,
        0xDA,
        0x8E,  # 3
        0x96,
        0x8F,
        0xDB,
        0xBD,
        0x36,
        0xD0,
        0xCE,
        0x94,
        0x13,
        0x5C,
        0xD2,
        0xF1,
        0x40,
        0x46,
        0x83,
        0x38,  # 4
        0x66,
        0xDD,
        0xFD,
        0x30,
        0xBF,
        0x06,
        0x8B,
        0x62,
        0xB3,
        0x25,
        0xE2,
        0x98,
        0x22,
        0x88,
        0x91,
        0x10,  # 5
        0x7E,
        0x6E,
        0x48,
        0xC3,
        0xA3,
        0xB6,
        0x1E,
        0x42,
        0x3A,
        0x6B,
        0x28,
        0x54,
        0xFA,
        0x85,
        0x3D,
        0xBA,  # 6
        0x2B,
        0x79,
        0x0A,
        0x15,
        0x9B,
        0x9F,
        0x5E,
        0xCA,
        0x4E,
        0xD4,
        0xAC,
        0xE5,
        0xF3,
        0x73,
        0xA7,
        0x57,  # 7
        0xAF,
        0x58,
        0xA8,
        0x50,
        0xF4,
        0xEA,
        0xD6,
        0x74,
        0x4F,
        0xAE,
        0xE9,
        0xD5,
        0xE7,
        0xE6,
        0xAD,
        0xE8,  # 8
        0x2C,
        0xD7,
        0x75,
        0x7A,
        0xEB,
        0x16,
        0x0B,
        0xF5,
        0x59,
        0xCB,
        0x5F,
        0xB0,
        0x9C,
        0xA9,
        0x51,
        0xA0,  # 9
        0x7F,
        0x0C,
        0xF6,
        0x6F,
        0x17,
        0xC4,
        0x49,
        0xEC,
        0xD8,
        0x43,
        0x1F,
        0x2D,
        0xA4,
        0x76,
        0x7B,
        0xB7,  # a
        0xCC,
        0xBB,
        0x3E,
        0x5A,
        0xFB,
        0x60,
        0xB1,
        0x86,
        0x3B,
        0x52,
        0xA1,
        0x6C,
        0xAA,
        0x55,
        0x29,
        0x9D,  # b
        0x97,
        0xB2,
        0x87,
        0x90,
        0x61,
        0xBE,
        0xDC,
        0xFC,
        0xBC,
        0x95,
        0xCF,
        0xCD,
        0x37,
        0x3F,
        0x5B,
        0xD1,  # c
        0x53,
        0x39,
        0x84,
        0x3C,
        0x41,
        0xA2,
        0x6D,
        0x47,
        0x14,
        0x2A,
        0x9E,
        0x5D,
        0x56,
        0xF2,
        0xD3,
        0xAB,  # d
        0x44,
        0x11,
        0x92,
        0xD9,
        0x23,
        0x20,
        0x2E,
        0x89,
        0xB4,
        0x7C,
        0xB8,
        0x26,
        0x77,
        0x99,
        0xE3,
        0xA5,  # e
        0x67,
        0x4A,
        0xED,
        0xDE,
        0xC5,
        0x31,
        0xFE,
        0x18,
        0x0D,
        0x63,
        0x8C,
        0x80,
        0xC0,
        0xF7,
        0x70,
        0x07,  # f
    ],
    dtype=int16,
)


def gf_multiply(x: UInt8Array, y: UInt8Array, _ff: int = 0xFF) -> UInt8Array:
    """Vectorized multiplication in GF(2^8).

    Note: this accepts arrays only, not scalars.  Otherwise the
    assignment on the last line will fail and we don't feel
    like making that check.
    """

    res = LTABLE[x] + LTABLE[y]
    res = ETABLE[np.where(res > 0xFF, res - _ff, res)]
    # Any number multiplied by zero GF(2^8) equals zero
    res[~np.logical_and(x, y)] = 0
    return res


def _mix_columns(
    state: UInt8Array,
    out: UInt8Array | None = None,
    *,
    pm0: UInt8Array,
    pm1: UInt8Array,
    pm2: UInt8Array,
    pm3: UInt8Array,
) -> UInt8Array:
    if out is not None:
        out[:] = (
            gf_multiply(state[0], pm0)
            ^ gf_multiply(state[1], pm1)
            ^ gf_multiply(state[2], pm2)
            ^ gf_multiply(state[3], pm3)
        )
        return out
    return (
        gf_multiply(state[0], pm0)
        ^ gf_multiply(state[1], pm1)
        ^ gf_multiply(state[2], pm2)
        ^ gf_multiply(state[3], pm3)
    )


ax_polynomial = array(
    [
        [0x02, 0x03, 0x01, 0x01],
        [0x01, 0x02, 0x03, 0x01],
        [0x01, 0x01, 0x02, 0x03],
        [0x03, 0x01, 0x01, 0x02],
    ],
    dtype=uint8,
)

pm0, pm1, pm2, pm3 = (
    ax_polynomial[:, [0]],
    ax_polynomial[:, [1]],
    ax_polynomial[:, [2]],
    ax_polynomial[:, [3]],
)

inv_ax_polynomial = array(
    [
        [0x0E, 0x0B, 0x0D, 0x09],
        [0x09, 0x0E, 0x0B, 0x0D],
        [0x0D, 0x09, 0x0E, 0x0B],
        [0x0B, 0x0D, 0x09, 0x0E],
    ],
    dtype=uint8,
)

ipm0, ipm1, ipm2, ipm3 = (
    inv_ax_polynomial[:, [0]],
    inv_ax_polynomial[:, [1]],
    inv_ax_polynomial[:, [2]],
    inv_ax_polynomial[:, [3]],
)

mix_columns = functools.partial(_mix_columns, pm0=pm0, pm1=pm1, pm2=pm2, pm3=pm3)
inv_mix_columns = functools.partial(_mix_columns, pm0=ipm0, pm1=ipm1, pm2=ipm2, pm3=ipm3)


def rot_word(word: UInt8Array) -> UInt8Array:
    """Takes a 4-byte word and performs cyclic permutation.

    Aka one-byte left circular shift.

    [b0, b1, b2, b3] -> [b1, b2, b3, b0]
    """
    return word[[1, 2, 3, 0]]


# "A function that takes a four-byte input word and applies the S-box
# to each of the four bytes to produce an output word"
sub_word = sub_bytes

# "The round constant word array, Rcon[i], contains the values
# given by [xi-1,{00},{00},{00}], with x i-1 being powers of
# x (x is denoted as {02}) in the field GF(2^8)"
# (I.e. powers of X % polynomial in GF(2^8))
#
# Just a tuple for now, because tuple access is about 4 times as fast
# as ndarray access, at least for this size & type
#
# Note: this is 1-indexed, hence empty first row
#
# Generated via:
#
# for i in range(1, 16):
#     if i == 1:
#         j = 1
#     elif i > 1 and j < 0x80:
#         j = 2 * j
#     elif i > 1 and j >= 0x80:
#         j = (2 * j) ^ 0x11b
#     yield j
RCON = (
    array([np.nan, np.nan, np.nan, np.nan]),  # NAN
    array([0x01, 0x00, 0x00, 0x00], dtype=uint8),  # 1
    array([0x02, 0x00, 0x00, 0x00], dtype=uint8),  # 2
    array([0x04, 0x00, 0x00, 0x00], dtype=uint8),  # 4
    array([0x08, 0x00, 0x00, 0x00], dtype=uint8),  # 8
    array([0x10, 0x00, 0x00, 0x00], dtype=uint8),  # 16
    array([0x20, 0x00, 0x00, 0x00], dtype=uint8),  # 32
    array([0x40, 0x00, 0x00, 0x00], dtype=uint8),  # 64
    array([0x80, 0x00, 0x00, 0x00], dtype=uint8),  # 128
    array([0x1B, 0x00, 0x00, 0x00], dtype=uint8),  # 27
    array([0x36, 0x00, 0x00, 0x00], dtype=uint8),  # 54
    array([0x6C, 0x00, 0x00, 0x00], dtype=uint8),  # 108
    array([0xD8, 0x00, 0x00, 0x00], dtype=uint8),  # 216
    array([0xAB, 0x00, 0x00, 0x00], dtype=uint8),  # 171
    array([0x4D, 0x00, 0x00, 0x00], dtype=uint8),  # 77
    array([0x9A, 0x00, 0x00, 0x00], dtype=uint8),  # 154
)


def expand_key(key: UInt8Array) -> UInt8Array:
    """Key expansion routine to generate a key schedule.

    Expand a short key into a single array of round keys, each 4 words.

    The key expansion generates a total of Nb * (Nr + 1) 4-byte words:
    - 128 -> 4 * (10 + 1) -> 44
    - 192 -> 4 * (12 + 1) -> 52
    - 256 -> 4 * (14 + 1) -> 60

    This diverges from the specification in that they expanded key is
    *not* a parameter that can be modified and remember its state.
    Rather, it gets created here as an empty array and then filled.
    """

    # The first *nk* words of the expanded key are filled with
    # respective keys from the cipher key.  (Each word == 4 bytes)
    nk = cast(Nk, int(key.size / 4))
    nr = numrounds(nk)
    nwords = NB * (nr + 1)
    w = np.empty((NB, nwords), dtype=uint8)
    w[:, :nk] = key
    # Now fill out the rest of w.
    # The dreaded loop:
    # It may be near-impossible to do this without a loop, since
    # we're dependent on the value w[i - nk]
    i = nk
    while i < nwords:
        # Starts on row 3
        temp = w[:, i - 1]
        if i % nk == 0:
            temp = xor(
                # XOR of a 4-byte word with length-4 lookup from RCON table
                sub_word(rot_word(temp)),
                RCON[int(i / nk)],
            )
        elif nk > 6 and i % nk == 4:
            temp = sub_word(temp)
        w[:, i] = xor(w[:, i - nk], temp)
        i += 1
    return w


def encrypt_raw(state: UInt8Array, key: UInt8Array) -> UInt8Array:
    """Encrypt a single input data block, `state`, using `key`.

    Modifies `state` in-place!

    Parameters
    ----------
    inp: np.ndarray
    key: np.ndarray

    Returns
    -------
    state: np.ndarray
    """

    # We get the keys in "human form" and apply AES' column-based axis swap on them
    # (This should happen within `encrypt()`)
    # Retain the 'original' key for our expand_key methodology
    # key = key.swapaxes(0, 1)
    nk = cast(Nk, int(key.size / 4))
    nr = numrounds(nk)
    exkeys = expand_key(key)  # 4 rows, NB * (nr + 1) columns (words)
    exkeys = np.split(exkeys, exkeys.shape[1] / NB, axis=1)

    # First XOR is with just input + key
    xor(state, exkeys[0], out=state)

    # Intermediate rounds
    for ek in exkeys[1:nr]:
        sub_bytes(state, out=state)
        shift_rows(state, out=state)
        mix_columns(state, out=state)
        xor(state, ek, out=state)

    # Final round before a final XOR.  No mixColumns here
    sub_bytes(state, out=state)
    shift_rows(state, out=state)
    xor(state, exkeys[-1], out=state)
    # TODO: this is returned in column-ordered format.
    # We will need to massage it back into row ordered before flattening
    return state
    # return bytes(state.flat)


# ---------------------------------------------------------------------
# Core decryption functions:
# InvSubBytes(), InvShiftRows(), InvMixColumns()
# AddRoundKey() is its own inverse

invcolindexer = array([[0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1], [1, 2, 3, 0]], dtype=uint8)


def inv_shift_rows(
    state: UInt8Array,
    out: UInt8Array | None = None,
    _rows=arange(4, dtype=uint8)[:, None],
    _cols=invcolindexer,
) -> UInt8Array:
    """Cyclically shift last 3 rows in the State, inverse."""
    if out is not None:
        out[:] = state[_rows, _cols]
        return out
    return state[_rows, _cols]


INVSBOX = array(
    [
        #  0     1     2     3     4     5     6     7     8     9     a     b     c     d     e     f
        0x52,
        0x09,
        0x6A,
        0xD5,
        0x30,
        0x36,
        0xA5,
        0x38,
        0xBF,
        0x40,
        0xA3,
        0x9E,
        0x81,
        0xF3,
        0xD7,
        0xFB,  # 0
        0x7C,
        0xE3,
        0x39,
        0x82,
        0x9B,
        0x2F,
        0xFF,
        0x87,
        0x34,
        0x8E,
        0x43,
        0x44,
        0xC4,
        0xDE,
        0xE9,
        0xCB,  # 1
        0x54,
        0x7B,
        0x94,
        0x32,
        0xA6,
        0xC2,
        0x23,
        0x3D,
        0xEE,
        0x4C,
        0x95,
        0x0B,
        0x42,
        0xFA,
        0xC3,
        0x4E,  # 2
        0x08,
        0x2E,
        0xA1,
        0x66,
        0x28,
        0xD9,
        0x24,
        0xB2,
        0x76,
        0x5B,
        0xA2,
        0x49,
        0x6D,
        0x8B,
        0xD1,
        0x25,  # 3
        0x72,
        0xF8,
        0xF6,
        0x64,
        0x86,
        0x68,
        0x98,
        0x16,
        0xD4,
        0xA4,
        0x5C,
        0xCC,
        0x5D,
        0x65,
        0xB6,
        0x92,  # 4
        0x6C,
        0x70,
        0x48,
        0x50,
        0xFD,
        0xED,
        0xB9,
        0xDA,
        0x5E,
        0x15,
        0x46,
        0x57,
        0xA7,
        0x8D,
        0x9D,
        0x84,  # 5
        0x90,
        0xD8,
        0xAB,
        0x00,
        0x8C,
        0xBC,
        0xD3,
        0x0A,
        0xF7,
        0xE4,
        0x58,
        0x05,
        0xB8,
        0xB3,
        0x45,
        0x06,  # 6
        0xD0,
        0x2C,
        0x1E,
        0x8F,
        0xCA,
        0x3F,
        0x0F,
        0x02,
        0xC1,
        0xAF,
        0xBD,
        0x03,
        0x01,
        0x13,
        0x8A,
        0x6B,  # 7
        0x3A,
        0x91,
        0x11,
        0x41,
        0x4F,
        0x67,
        0xDC,
        0xEA,
        0x97,
        0xF2,
        0xCF,
        0xCE,
        0xF0,
        0xB4,
        0xE6,
        0x73,  # 8
        0x96,
        0xAC,
        0x74,
        0x22,
        0xE7,
        0xAD,
        0x35,
        0x85,
        0xE2,
        0xF9,
        0x37,
        0xE8,
        0x1C,
        0x75,
        0xDF,
        0x6E,  # 9
        0x47,
        0xF1,
        0x1A,
        0x71,
        0x1D,
        0x29,
        0xC5,
        0x89,
        0x6F,
        0xB7,
        0x62,
        0x0E,
        0xAA,
        0x18,
        0xBE,
        0x1B,  # a
        0xFC,
        0x56,
        0x3E,
        0x4B,
        0xC6,
        0xD2,
        0x79,
        0x20,
        0x9A,
        0xDB,
        0xC0,
        0xFE,
        0x78,
        0xCD,
        0x5A,
        0xF4,  # b
        0x1F,
        0xDD,
        0xA8,
        0x33,
        0x88,
        0x07,
        0xC7,
        0x31,
        0xB1,
        0x12,
        0x10,
        0x59,
        0x27,
        0x80,
        0xEC,
        0x5F,  # c
        0x60,
        0x51,
        0x7F,
        0xA9,
        0x19,
        0xB5,
        0x4A,
        0x0D,
        0x2D,
        0xE5,
        0x7A,
        0x9F,
        0x93,
        0xC9,
        0x9C,
        0xEF,  # d
        0xA0,
        0xE0,
        0x3B,
        0x4D,
        0xAE,
        0x2A,
        0xF5,
        0xB0,
        0xC8,
        0xEB,
        0xBB,
        0x3C,
        0x83,
        0x53,
        0x99,
        0x61,  # e
        0x17,
        0x2B,
        0x04,
        0x7E,
        0xBA,
        0x77,
        0xD6,
        0x26,
        0xE1,
        0x69,
        0x14,
        0x63,
        0x55,
        0x21,
        0x0C,
        0x7D,  # f
    ],
    dtype=uint8,
)


def inv_sub_bytes(state: UInt8Array, out: UInt8Array | None = None, _sbox=INVSBOX) -> UInt8Array:
    if out is not None:
        out[:] = _sbox[state]
        return out
    return _sbox[state]


def decrypt_raw(state: UInt8Array, key: UInt8Array) -> UInt8Array:
    """Decrypt a single ciphertext data block, `state`, using `key`.

    Modifies `state` in-place!

    Parameters
    ----------
    inp: np.ndarray
    key: np.ndarray

    Returns
    -------
    state: np.ndarray
    """
    nk = cast(Nk, int(key.size / 4))
    nr = numrounds(nk)
    exkeys = expand_key(key)  # 4 rows, NB * (nr + 1) columns (words)
    exkeys = np.split(exkeys, exkeys.shape[1] / NB, axis=1)

    # First XOR is with just input + key (reverse order of roundkeys)
    # Last round doesn't get an InvMixColumns
    xor(state, exkeys[-1], out=state)
    inv_shift_rows(state, out=state)
    inv_sub_bytes(state, out=state)

    for ek in exkeys[nr - 1 : 0 : -1]:
        xor(state, ek, out=state)
        inv_mix_columns(state, out=state)
        inv_shift_rows(state, out=state)
        inv_sub_bytes(state, out=state)

    # One final (inverse) xor
    xor(state, exkeys[0], out=state)
    return state


# ---------------------------------------------------------------------
# Helpers
# These are really only used in test_npaes.py for round-trip tests
# of the copied-over example vectors, which are in hex format.
# (And from a PDF, to boot.)


def hex_to_array(s: str, ndim: Literal[1, 2] = 2) -> UInt8Array:
    """Produce an array of uint8 bytes from a hex string.

    If ndim is 1, the result is 1d, length 16.
    If ndim is 2, the result is 4x4 and follows FIPS197's
    *column ordering*, with each column denoting a successive
    block from the input.

    Example
    -------
    >>> hex_to_array("00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f")
    array([[ 0,  4,  8, 12],
           [ 1,  5,  9, 13],
           [ 2,  6, 10, 14],
           [ 3,  7, 11, 15]])

    >>> hex_to_array("00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f", 1)
    >>> array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
    """

    # bytearray, not bytes, is Py 2-3 compatible
    res = array(bytearray.fromhex(s), dtype=uint8)
    if ndim == 2:
        size = int(len(res) / 4)
        res = res.reshape(size, 4).swapaxes(0, 1)
    return res


def array_to_hex(arr: UInt8Array, sep: str = " ") -> str:
    """Inverse of `hex_to_array()`."""
    # Or: binascii.hexlify(bytearray(out.swapaxes(0, 1).flat)).decode("ascii")
    if arr.ndim == 1:
        return sep.join(map("{:02x}".format, arr))
    return sep.join(map("{:02x}".format, arr.swapaxes(0, 1).flat))


def array_to_bytes(arr: UInt8Array) -> bytes:
    return bytes(arr.swapaxes(0, 1).flat)


# Note: don't use np.frombuffer() here if tempted.  It returns read-only


def plaintext_to_3darray(b: bytes) -> UInt8Array:
    whole = array(bytearray(b), dtype=np.uint8).reshape(-1, 4)
    return as_strided(whole, (int(whole.size / BLOCKSIZE_BYTES), 4, 4)).swapaxes(1, 2)


def key_to_array(key: bytes) -> UInt8Array:
    return array(bytearray(key), dtype=np.uint8).reshape(-1, 4).swapaxes(0, 1)
