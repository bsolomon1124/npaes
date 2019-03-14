# -*- coding: utf-8 -*-

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

 - Internally, the AES algorithm’s operations are performed on a
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

from __future__ import division
from __future__ import unicode_literals

__all__ = ()
__version__ = "0.1"

import functools

import numpy as np
from numpy import arange, array, uint8, int16, bitwise_xor as xor

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


def numrounds(nk, _table={4: 10, 6: 12, 8: 14}):
    """Return number of rounds (Nr) as function of key size (Nk).

    It is technically a function of Nk and Nb, but Nb is fixed at 4.
    """
    return _table[nk]


# ---------------------------------------------------------------------
# Core encryption functions:
# SubBytes(), ShiftRows(), MixColumns()
# (AddRoundKey() is just an XOR)

SBOX = array([
    #  0     1     2     3     4     5     6     7     8     9     a     b     c     d     e     f
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,  # 0
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,  # 1
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,  # 2
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,  # 3
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,  # 4
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,  # 5
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,  # 6
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,  # 7
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,  # 8
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,  # 9
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,  # a
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,  # b
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,  # c
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,  # d
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,  # e
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,  # f
], dtype=uint8)


def sub_bytes(state, out=None, _sbox=SBOX):
    if out is not None:
        out[:] = _sbox[state]
        return out
    else:
        return _sbox[state]


# Cols is also
# np.atleast_2d(arange(4)) - array([0, 3, 2, 1])[:, None]
# ...but that doesn't seem any simpler now, does it?
colindexer = array([[0, 1, 2, 3],
                    [1, 2, 3, 0],
                    [2, 3, 0, 1],
                    [3, 0, 1, 2]], dtype=uint8)


def shift_rows(
    state,
    out=None,
    _rows=arange(4, dtype=uint8)[:, None],
    _cols=colindexer,
):
    """Cyclically shift last 3 rows in the State."""
    if out is not None:
        out[:] = state[_rows, _cols]
        return out
    else:
        return state[_rows, _cols]


# ETABLE and LTABLE are lookup tables for matrix multiplication in GF(2^8)
ETABLE = array([
    #  0     1     2     3     4     5     6     7     8     9     a     b     c     d     e     f
    0x01, 0x03, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x2E, 0x72, 0x96, 0xA1, 0xF8, 0x13, 0x35,  # 0
    0x5F, 0xE1, 0x38, 0x48, 0xD8, 0x73, 0x95, 0xA4, 0xF7, 0x02, 0x06, 0x0A, 0x1E, 0x22, 0x66, 0xAA,  # 1
    0xE5, 0x34, 0x5C, 0xE4, 0x37, 0x59, 0xEB, 0x26, 0x6A, 0xBE, 0xD9, 0x70, 0x90, 0xAB, 0xE6, 0x31,  # 2
    0x53, 0xF5, 0x04, 0x0C, 0x14, 0x3C, 0x44, 0xCC, 0x4F, 0xD1, 0x68, 0xB8, 0xD3, 0x6E, 0xB2, 0xCD,  # 3
    0x4C, 0xD4, 0x67, 0xA9, 0xE0, 0x3B, 0x4D, 0xD7, 0x62, 0xA6, 0xF1, 0x08, 0x18, 0x28, 0x78, 0x88,  # 4
    0x83, 0x9E, 0xB9, 0xD0, 0x6B, 0xBD, 0xDC, 0x7F, 0x81, 0x98, 0xB3, 0xCE, 0x49, 0xDB, 0x76, 0x9A,  # 5
    0xB5, 0xC4, 0x57, 0xF9, 0x10, 0x30, 0x50, 0xF0, 0x0B, 0x1D, 0x27, 0x69, 0xBB, 0xD6, 0x61, 0xA3,  # 6
    0xFE, 0x19, 0x2B, 0x7D, 0x87, 0x92, 0xAD, 0xEC, 0x2F, 0x71, 0x93, 0xAE, 0xE9, 0x20, 0x60, 0xA0,  # 7
    0xFB, 0x16, 0x3A, 0x4E, 0xD2, 0x6D, 0xB7, 0xC2, 0x5D, 0xE7, 0x32, 0x56, 0xFA, 0x15, 0x3F, 0x41,  # 8
    0xC3, 0x5E, 0xE2, 0x3D, 0x47, 0xC9, 0x40, 0xC0, 0x5B, 0xED, 0x2C, 0x74, 0x9C, 0xBF, 0xDA, 0x75,  # 9
    0x9F, 0xBA, 0xD5, 0x64, 0xAC, 0xEF, 0x2A, 0x7E, 0x82, 0x9D, 0xBC, 0xDF, 0x7A, 0x8E, 0x89, 0x80,  # a
    0x9B, 0xB6, 0xC1, 0x58, 0xE8, 0x23, 0x65, 0xAF, 0xEA, 0x25, 0x6F, 0xB1, 0xC8, 0x43, 0xC5, 0x54,  # b
    0xFC, 0x1F, 0x21, 0x63, 0xA5, 0xF4, 0x07, 0x09, 0x1B, 0x2D, 0x77, 0x99, 0xB0, 0xCB, 0x46, 0xCA,  # c
    0x45, 0xCF, 0x4A, 0xDE, 0x79, 0x8B, 0x86, 0x91, 0xA8, 0xE3, 0x3E, 0x42, 0xC6, 0x51, 0xF3, 0x0E,  # d
    0x12, 0x36, 0x5A, 0xEE, 0x29, 0x7B, 0x8D, 0x8C, 0x8F, 0x8A, 0x85, 0x94, 0xA7, 0xF2, 0x0D, 0x17,  # e
    0x39, 0x4B, 0xDD, 0x7C, 0x84, 0x97, 0xA2, 0xFD, 0x1C, 0x24, 0x6C, 0xB4, 0xC7, 0x52, 0xF6, 0x01,  # f
], dtype=uint8)

# There is no value at 0,0 here, just a placeholder.  dtype is int16, not
# uint8, because we will be adding two of these together
LTABLE = array([
    #  0     1     2     3     4     5     6     7     8     9     a     b     c     d     e     f
    0x00, 0x00, 0x19, 0x01, 0x32, 0x02, 0x1A, 0xC6, 0x4B, 0xC7, 0x1B, 0x68, 0x33, 0xEE, 0xDF, 0x03,  # 0
    0x64, 0x04, 0xE0, 0x0E, 0x34, 0x8D, 0x81, 0xEF, 0x4C, 0x71, 0x08, 0xC8, 0xF8, 0x69, 0x1C, 0xC1,  # 1
    0x7D, 0xC2, 0x1D, 0xB5, 0xF9, 0xB9, 0x27, 0x6A, 0x4D, 0xE4, 0xA6, 0x72, 0x9A, 0xC9, 0x09, 0x78,  # 2
    0x65, 0x2F, 0x8A, 0x05, 0x21, 0x0F, 0xE1, 0x24, 0x12, 0xF0, 0x82, 0x45, 0x35, 0x93, 0xDA, 0x8E,  # 3
    0x96, 0x8F, 0xDB, 0xBD, 0x36, 0xD0, 0xCE, 0x94, 0x13, 0x5C, 0xD2, 0xF1, 0x40, 0x46, 0x83, 0x38,  # 4
    0x66, 0xDD, 0xFD, 0x30, 0xBF, 0x06, 0x8B, 0x62, 0xB3, 0x25, 0xE2, 0x98, 0x22, 0x88, 0x91, 0x10,  # 5
    0x7E, 0x6E, 0x48, 0xC3, 0xA3, 0xB6, 0x1E, 0x42, 0x3A, 0x6B, 0x28, 0x54, 0xFA, 0x85, 0x3D, 0xBA,  # 6
    0x2B, 0x79, 0x0A, 0x15, 0x9B, 0x9F, 0x5E, 0xCA, 0x4E, 0xD4, 0xAC, 0xE5, 0xF3, 0x73, 0xA7, 0x57,  # 7
    0xAF, 0x58, 0xA8, 0x50, 0xF4, 0xEA, 0xD6, 0x74, 0x4F, 0xAE, 0xE9, 0xD5, 0xE7, 0xE6, 0xAD, 0xE8,  # 8
    0x2C, 0xD7, 0x75, 0x7A, 0xEB, 0x16, 0x0B, 0xF5, 0x59, 0xCB, 0x5F, 0xB0, 0x9C, 0xA9, 0x51, 0xA0,  # 9
    0x7F, 0x0C, 0xF6, 0x6F, 0x17, 0xC4, 0x49, 0xEC, 0xD8, 0x43, 0x1F, 0x2D, 0xA4, 0x76, 0x7B, 0xB7,  # a
    0xCC, 0xBB, 0x3E, 0x5A, 0xFB, 0x60, 0xB1, 0x86, 0x3B, 0x52, 0xA1, 0x6C, 0xAA, 0x55, 0x29, 0x9D,  # b
    0x97, 0xB2, 0x87, 0x90, 0x61, 0xBE, 0xDC, 0xFC, 0xBC, 0x95, 0xCF, 0xCD, 0x37, 0x3F, 0x5B, 0xD1,  # c
    0x53, 0x39, 0x84, 0x3C, 0x41, 0xA2, 0x6D, 0x47, 0x14, 0x2A, 0x9E, 0x5D, 0x56, 0xF2, 0xD3, 0xAB,  # d
    0x44, 0x11, 0x92, 0xD9, 0x23, 0x20, 0x2E, 0x89, 0xB4, 0x7C, 0xB8, 0x26, 0x77, 0x99, 0xE3, 0xA5,  # e
    0x67, 0x4A, 0xED, 0xDE, 0xC5, 0x31, 0xFE, 0x18, 0x0D, 0x63, 0x8C, 0x80, 0xC0, 0xF7, 0x70, 0x07,  # f
], dtype=int16)


def gf_multiply(x, y, _ff=np.int(0xff)):
    """Vectorized multiplication in GF(2^8).

    Note: this accepts arrays only, not scalars.  Otherwise the
    assignment on the last line will fail and we don't feel
    like making that check.
    """

    res = LTABLE[x] + LTABLE[y]
    res = ETABLE[np.where(res > 0xff, res - _ff, res)]
    # Any number multiplied by zero GF(2^8) equals zero
    res[~np.logical_and(x, y)] = 0
    return res


def _mix_columns(state, out=None, pm0=None, pm1=None, pm2=None, pm3=None):
    if out is not None:
        out[:] = gf_multiply(state[0], pm0) ^ \
                 gf_multiply(state[1], pm1) ^ \
                 gf_multiply(state[2], pm2) ^ \
                 gf_multiply(state[3], pm3)  # noqa
        return out
    else:
        return gf_multiply(state[0], pm0) ^ \
               gf_multiply(state[1], pm1) ^ \
               gf_multiply(state[2], pm2) ^ \
               gf_multiply(state[3], pm3)  # noqa


ax_polynomial = array(
    [
        [0x02, 0x03, 0x01, 0x01],
        [0x01, 0x02, 0x03, 0x01],
        [0x01, 0x01, 0x02, 0x03],
        [0x03, 0x01, 0x01, 0x02],
    ], dtype=uint8)

pm0, pm1, pm2, pm3 = (
    ax_polynomial[:, [0]],
    ax_polynomial[:, [1]],
    ax_polynomial[:, [2]],
    ax_polynomial[:, [3]],
)

inv_ax_polynomial = array(
    [
        [0x0e, 0x0b, 0x0d, 0x09],
        [0x09, 0x0e, 0x0b, 0x0d],
        [0x0d, 0x09, 0x0e, 0x0b],
        [0x0b, 0x0d, 0x09, 0x0e],
    ], dtype=uint8)

ipm0, ipm1, ipm2, ipm3 = (
    inv_ax_polynomial[:, [0]],
    inv_ax_polynomial[:, [1]],
    inv_ax_polynomial[:, [2]],
    inv_ax_polynomial[:, [3]],
)

mix_columns = functools.partial(
    _mix_columns,
    pm0=pm0, pm1=pm1, pm2=pm2, pm3=pm3
)
inv_mix_columns = functools.partial(
    _mix_columns,
    pm0=ipm0, pm1=ipm1, pm2=ipm2, pm3=ipm3
)


def rot_word(word):
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
    array([0x6c, 0x00, 0x00, 0x00], dtype=uint8),  # 108
    array([0xd8, 0x00, 0x00, 0x00], dtype=uint8),  # 216
    array([0xab, 0x00, 0x00, 0x00], dtype=uint8),  # 171
    array([0x4d, 0x00, 0x00, 0x00], dtype=uint8),  # 77
    array([0x9a, 0x00, 0x00, 0x00], dtype=uint8),  # 154
)


def expand_key(key):
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
    nk = int(key.size / 4)
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
                RCON[int(i / nk)]
            )
        elif nk > 6 and i % nk == 4:
            temp = sub_word(temp)
        w[:, i] = xor(w[:, i - nk], temp)
        i += 1
    return w


def encrypt_raw(state, key):
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
    nk = int(key.size / 4)
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

invcolindexer = array([[0, 1, 2, 3],
                       [3, 0, 1, 2],
                       [2, 3, 0, 1],
                       [1, 2, 3, 0]], dtype=uint8)


def inv_shift_rows(
    state,
    out=None,
    _rows=arange(4, dtype=uint8)[:, None],
    _cols=invcolindexer,
):
    """Cyclically shift last 3 rows in the State, inverse."""
    if out is not None:
        out[:] = state[_rows, _cols]
        return out
    else:
        return state[_rows, _cols]


INVSBOX = array([
    #  0     1     2     3     4     5     6     7     8     9     a     b     c     d     e     f
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,  # 0
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,  # 1
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,  # 2
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,  # 3
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,  # 4
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,  # 5
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,  # 6
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,  # 7
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,  # 8
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,  # 9
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,  # a
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,  # b
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,  # c
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,  # d
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,  # e
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,  # f
], dtype=uint8)


def inv_sub_bytes(state, out=None, _sbox=INVSBOX):
    if out is not None:
        out[:] = _sbox[state]
        return out
    else:
        return _sbox[state]


def decrypt_raw(state, key):
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
    nk = int(key.size / 4)
    nr = numrounds(nk)
    exkeys = expand_key(key)  # 4 rows, NB * (nr + 1) columns (words)
    exkeys = np.split(exkeys, exkeys.shape[1] / NB, axis=1)

    # First XOR is with just input + key (reverse order of roundkeys)
    # Last round doesn't get an InvMixColumns
    xor(state, exkeys[-1], out=state)
    inv_shift_rows(state, out=state)
    inv_sub_bytes(state, out=state)

    for ek in exkeys[nr - 1:0:-1]:
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


def hex_to_array(s, ndim=2):
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


def array_to_hex(arr, sep=" "):
    """Inverse of `hex_to_array()`."""
    # Or: binascii.hexlify(bytearray(out.swapaxes(0, 1).flat)).decode("ascii")
    if arr.ndim == 1:
        return sep.join(map("{:02x}".format, arr))
    return sep.join(map("{:02x}".format, arr.swapaxes(0, 1).flat))
