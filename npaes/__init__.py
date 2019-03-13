# -*- coding: utf-8 -*-

"""Advanced Encryption Standard (AES) NumPy implementation.

Based strictly on:

    Federal Information Processing Standards Publication 197
    November 26, 2001
    https://csrc.nist.gov/publications/detail/fips/197/final

Any reference to the paper in this source code is called just FIPS197.

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

 - One last technical note is that, because we're working with bits,
   we work almost exclusively with np.uint8 dtype.
"""

from __future__ import division
from __future__ import unicode_literals

__all__ = ("encrypt_raw", )
__version__ = "0.1"

import numpy as np
from numpy import arange, array, uint8, bitwise_xor as xor

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
# Core Cipher functions:
# SubBytes(), ShiftRows(), MixColumns(), and AddRoundKey()

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


def sub_bytes(state, _sbox=SBOX):
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
    _rows=arange(4, dtype=uint8)[:, None],
    _cols=colindexer,
):
    """Cyclically shift last 3 rows in the State."""
    return state[_rows, _cols]


ax_polynomial = array(
    [
        [0x02, 0x03, 0x01, 0x01],
        [0x01, 0x02, 0x03, 0x01],
        [0x01, 0x01, 0x02, 0x03],
        [0x03, 0x01, 0x01, 0x02],
    ], dtype=uint8)  # uint8 okay?


def mix_columns(state):
    """MixColumns operation, primary source of diffusion.

    'The MixColumns() transformation operates on the State
    column-by-column, treating each column as a four-term polynomial.'

    Here is the transformation of 1 column (underscore means subscript):

       [s_0c] = [2, 3, 1, 1][s_0c]
       [s_1c] = [1, 2, 3, 1][s_1c]
       [s_2c] = [1, 1, 2, 3][s_2c]
       [s_3c] = [3, 1, 1, 2][s_3c]

    Note that this is *not* a normaal dot product.  It is:

        s_0c = (0x02 • s_1c) ⊕ (0x03 • s_1c) ⊕ (0x01 • s_2c) ⊕ (0x01 • s_3c)
        s_0c = (0x01 • s_1c) ⊕ (0x02 • s_1c) ⊕ (0x03 • s_2c) ⊕ (0x01 • s_3c)
        s_0c = (0x01 • s_1c) ⊕ (0x01 • s_1c) ⊕ (0x02 • s_2c) ⊕ (0x03 • s_3c)
        s_0c = (0x03 • s_1c) ⊕ (0x01 • s_1c) ⊕ (0x01 • s_2c) ⊕ (0x02 • s_3c)

    where the dot • denotes multiplication in GF(2^8), see section 4.2.

    Example:
    https://en.wikipedia.org/wiki/Rijndael_MixColumns#Implementation_example
    """

    assert state.dtype is np.dtype("uint8")
    h = np.where(state >> 7, 0xff, 0x00)
    b = state << 1
    # 0x1b (27 dec) comes from irreducible polynomial {0x01}{0x1b}
    b = b ^ 0x1b & h
    return b ^ state[[3, 0, 1, 2]] ^ state[[2, 3, 0, 1]] ^ b[[1, 2, 3, 0]] ^ state[[1, 2, 3, 0]]


# ---------------------------------------------------------------------


def rot_word(word):
    """Takes a 4-byte word and performs cyclic permutation.

    Aka one-byte left circular shift.

    [b0, b1, b2, b3] -> [b1, b2, b3, b0]
    """
    return word[[1, 2, 3, 0]]


# "A function that takes a four-byte input word and applies the S-box
# to each of the four bytes to produce an output word"
sub_word = sub_bytes


def key_to_words(key, nk):
    assert nk in ALLOWED_NK
    """Initial words from key key used in key expansion."""
    return np.split(key, nk)


# "The round constant word array, Rcon[i], contains the values
# given by [xi-1,{00},{00},{00}], with x i-1 being powers of
# x (x is denoted as {02}) in the field GF(28)"
#
# Just a tuple for now, because tuple access is about 4 times as fast
# as ndarray access, at least for this size & type
#
# Note: this is 1-indexed, hence empty first row
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

    Expand a short key into a single array of round keys.
    """

    key = key.flatten()
    nk = int(len(key) / 4)
    assert nk in ALLOWED_NK
    indexer = arange(len(key)).reshape(4, -1)

    # They key expansion generates a total of Nb * (Nr + 1) 4-byte words:
    # 128 -> 4 * (10 + 1) -> 44
    # 192 -> 4 * (12 + 1) -> 52
    # 256 -> 4 * (14 + 1) -> 60
    #
    # The first *nk* words of the expanded key are filled with
    # respective keys from the cipher key.  (Each word is 4 bytes)
    nr = numrounds(nk)  # 10
    nwords = NB * (nr + 1)
    w = np.empty((NB, nwords), dtype=uint8)
    w[:, :nk] = key[indexer]  # word 0 through word nk;
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
                sub_word(rot_word(temp)),
                RCON[int(i / nk)]
            )
        elif nk > 6 and i % nk == 4:
            temp = sub_word(temp)
        w[:, i] = xor(w[:, i - nk], temp)
        i += 1
    assert w.shape[1] == NB * (nr + 1)
    return w


def encrypt_raw(inp, key):
    """Encrypt a single input data block, `inp`, using `key`.

    Parameters
    ----------
    inp: np.ndarray
    key: np.ndarray

    Returns
    -------
    state: np.ndarray
    """

    assert inp.size == BLOCKSIZE_BYTES
    # We get the keys in "human form" and apply AES' column-based axis swap on them
    # (This should happen within `encrypt()`)
    # Retain the 'original' key for our expand_key methodology
    # key = key.swapaxes(0, 1)
    nk = int(key.size / 4)
    nr = numrounds(nk)  # 10
    exkeys = expand_key(key)  # 4 rows, NB * (nr + 1) columns (words)
    assert exkeys.shape[0] == NB
    exkeys = np.split(exkeys, exkeys.shape[1] / NB, axis=1)

    # First XOR is with just input + key
    state = xor(inp, exkeys[0])

    # Intermediate rounds
    for i in range(1, nr):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        xor(state, exkeys[i], out=state)

    # Final round before a final XOR.  No mixColumns here
    state = sub_bytes(state)
    state = shift_rows(state)
    state = xor(state, exkeys[-1])
    # TODO: this is returned in column-ordered format.
    # We will need to massage it back into row ordered before flattening
    return state
    # return bytes(state.flat)


# ---------------------------------------------------------------------
# Helpers


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
    return sep.join(map(hex, arr.flat))


def genrcon(upper=16):
    # [1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154]
    for i in range(1, upper):
        if i == 1:
            j = 1
        elif i > 1 and j < 0x80:
            j = 2 * j
        elif i > 1 and j >= 0x80:
            j = (2 * j) ^ 0x11b
        else:
            raise Exception("Condition check failed")
        yield j
