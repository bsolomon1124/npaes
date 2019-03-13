# /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import array, array_equal, uint8, bitwise_xor as xor
import pytest

from npaes import (
    array_to_hex,
    hex_to_array,
    mix_columns,
    shift_rows,
    sub_bytes,
    expand_key,
    encrypt_raw,
    RCON,
)

# ---------------------------------------------------------------------
# Utils and building blocks get tested first


def test_hex_to_array():
    s = "00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f"
    assert array_equal(
        hex_to_array(s, ndim=2),
        array([[0, 4, 8, 12],
               [1, 5, 9, 13],
               [2, 6, 10, 14],
               [3, 7, 11, 15]])
    )
    assert array_equal(
        hex_to_array(s, ndim=1),
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    )


def test_mix_columns():
    # A sliver of Appendix B, Round 1
    after_shift_rows = array(
        [
            [0xd4, 0xe0, 0xb8, 0x1e],
            [0xbf, 0xb4, 0x41, 0x27],
            [0x5d, 0x52, 0x11, 0x98],
            [0x30, 0xae, 0xf1, 0xe5],
        ], dtype=uint8)
    after_mix_columns = array(
        [
            [0x04, 0xe0, 0x48, 0x28],
            [0x66, 0xcb, 0xf8, 0x06],
            [0x81, 0x19, 0xd3, 0x26],
            [0xe5, 0x9a, 0x7a, 0x4c],
        ], dtype=uint8)
    assert array_equal(
        mix_columns(after_shift_rows),
        after_mix_columns
    )


def test_array_to_hex():
    assert array_to_hex(hex_to_array("2b7e1516", ndim=1)) == '0x2b 0x7e 0x15 0x16'
    assert array_to_hex(hex_to_array("e0 c8 d9 85 92 63 b1 b8 7f 63 35 be e8 c0 50 01", ndim=1)) == '0xe0 0xc8 0xd9 0x85 0x92 0x63 0xb1 0xb8 0x7f 0x63 0x35 0xbe 0xe8 0xc0 0x50 0x1'


# ---------------------------------------------------------------------
# Appendix A - Key Expansion Examples

# NOTE: Remember that FIPS197 is following column-ordering here.
# `w[i]` in the examples means the *column index* i.
# That means that the raw elements in the `*_true` arrays
# represent *columns* (hence why they get transposed below).
key128 = hex_to_array("2b 7e 15 16 28 ae d2 a6 ab f7 15 88 09 cf 4f 3c")
w128_true = array([
    hex_to_array("2b7e1516", ndim=1),  # 0
    hex_to_array("28aed2a6", ndim=1),  # 1
    hex_to_array("abf71588", ndim=1),  # 2
    hex_to_array("09cf4f3c", ndim=1),  # 3
    hex_to_array("a0fafe17", ndim=1),  # 4
    hex_to_array("88542cb1", ndim=1),  # 5
    hex_to_array("23a33939", ndim=1),  # 6
    hex_to_array("2a6c7605", ndim=1),  # 7
    hex_to_array("f2c295f2", ndim=1),  # 8
    hex_to_array("7a96b943", ndim=1),  # 9
    hex_to_array("5935807a", ndim=1),  # 10
    hex_to_array("7359f67f", ndim=1),  # 11
    hex_to_array("3d80477d", ndim=1),  # 12
    hex_to_array("4716fe3e", ndim=1),  # 13
    hex_to_array("1e237e44", ndim=1),  # 14
    hex_to_array("6d7a883b", ndim=1),  # 15
    hex_to_array("ef44a541", ndim=1),  # 16
    hex_to_array("a8525b7f", ndim=1),  # 17
    hex_to_array("b671253b", ndim=1),  # 18
    hex_to_array("db0bad00", ndim=1),  # 19
    hex_to_array("d4d1c6f8", ndim=1),  # 20
    hex_to_array("7c839d87", ndim=1),  # 21
    hex_to_array("caf2b8bc", ndim=1),  # 22
    hex_to_array("11f915bc", ndim=1),  # 23
    hex_to_array("6d88a37a", ndim=1),  # 24
    hex_to_array("110b3efd", ndim=1),  # 25
    hex_to_array("dbf98641", ndim=1),  # 26
    hex_to_array("ca0093fd", ndim=1),  # 27
    hex_to_array("4e54f70e", ndim=1),  # 28
    hex_to_array("5f5fc9f3", ndim=1),  # 29
    hex_to_array("84a64fb2", ndim=1),  # 30
    hex_to_array("4ea6dc4f", ndim=1),  # 31
    hex_to_array("ead27321", ndim=1),  # 32
    hex_to_array("b58dbad2", ndim=1),  # 33
    hex_to_array("312bf560", ndim=1),  # 34
    hex_to_array("7f8d292f", ndim=1),  # 35
    hex_to_array("ac7766f3", ndim=1),  # 36
    hex_to_array("19fadc21", ndim=1),  # 37
    hex_to_array("28d12941", ndim=1),  # 38
    hex_to_array("575c006e", ndim=1),  # 39
    hex_to_array("d014f9a8", ndim=1),  # 40
    hex_to_array("c9ee2589", ndim=1),  # 41
    hex_to_array("e13f0cc8", ndim=1),  # 42
    hex_to_array("b6630ca6", ndim=1),  # 43
]).T


key192 = hex_to_array(
    "8e 73 b0 f7 da 0e 64 52 c8 10 f3 2b"
    "80 90 79 e5 62 f8 ea d2 52 2c 6b 7b"
)
w192_true = array([
    hex_to_array("8e73b0f7", ndim=1),  # 0
    hex_to_array("da0e6452", ndim=1),  # 1
    hex_to_array("c810f32b", ndim=1),  # 2
    hex_to_array("809079e5", ndim=1),  # 3
    hex_to_array("62f8ead2", ndim=1),  # 4
    hex_to_array("522c6b7b", ndim=1),  # 5
    hex_to_array("fe0c91f7", ndim=1),  # 6
    hex_to_array("2402f5a5", ndim=1),  # 7
    hex_to_array("ec12068e", ndim=1),  # 8
    hex_to_array("6c827f6b", ndim=1),  # 9
    hex_to_array("0e7a95b9", ndim=1),  # 10
    hex_to_array("5c56fec2", ndim=1),  # 11
    hex_to_array("4db7b4bd", ndim=1),  # 12
    hex_to_array("69b54118", ndim=1),  # 13
    hex_to_array("85a74796", ndim=1),  # 14
    hex_to_array("e92538fd", ndim=1),  # 15
    hex_to_array("e75fad44", ndim=1),  # 16
    hex_to_array("bb095386", ndim=1),  # 17
    hex_to_array("485af057", ndim=1),  # 18
    hex_to_array("21efb14f", ndim=1),  # 19
    hex_to_array("a448f6d9", ndim=1),  # 20
    hex_to_array("4d6dce24", ndim=1),  # 21
    hex_to_array("aa326360", ndim=1),  # 22
    hex_to_array("113b30e6", ndim=1),  # 23
    hex_to_array("a25e7ed5", ndim=1),  # 24
    hex_to_array("83b1cf9a", ndim=1),  # 25
    hex_to_array("27f93943", ndim=1),  # 26
    hex_to_array("6a94f767", ndim=1),  # 27
    hex_to_array("c0a69407", ndim=1),  # 28
    hex_to_array("d19da4e1", ndim=1),  # 29
    hex_to_array("ec1786eb", ndim=1),  # 30
    hex_to_array("6fa64971", ndim=1),  # 31
    hex_to_array("485f7032", ndim=1),  # 32
    hex_to_array("22cb8755", ndim=1),  # 33
    hex_to_array("e26d1352", ndim=1),  # 34
    hex_to_array("33f0b7b3", ndim=1),  # 35
    hex_to_array("40beeb28", ndim=1),  # 36
    hex_to_array("2f18a259", ndim=1),  # 37
    hex_to_array("6747d26b", ndim=1),  # 38
    hex_to_array("458c553e", ndim=1),  # 39
    hex_to_array("a7e1466c", ndim=1),  # 40
    hex_to_array("9411f1df", ndim=1),  # 41
    hex_to_array("821f750a", ndim=1),  # 42
    hex_to_array("ad07d753", ndim=1),  # 43
    hex_to_array("ca400538", ndim=1),  # 44
    hex_to_array("8fcc5006", ndim=1),  # 45
    hex_to_array("282d166a", ndim=1),  # 46
    hex_to_array("bc3ce7b5", ndim=1),  # 47
    hex_to_array("e98ba06f", ndim=1),  # 48
    hex_to_array("448c773c", ndim=1),  # 49
    hex_to_array("8ecc7204", ndim=1),  # 50
    hex_to_array("01002202", ndim=1),  # 51
]).T


key256 = hex_to_array(
    "60 3d eb 10 15 ca 71 be 2b 73 ae f0 85 7d 77 81"
    "1f 35 2c 07 3b 61 08 d7 2d 98 10 a3 09 14 df f4",
)
w256_true = array([
    hex_to_array("603deb10", ndim=1),
    hex_to_array("15ca71be", ndim=1),
    hex_to_array("2b73aef0", ndim=1),
    hex_to_array("857d7781", ndim=1),
    hex_to_array("1f352c07", ndim=1),
    hex_to_array("3b6108d7", ndim=1),
    hex_to_array("2d9810a3", ndim=1),
    hex_to_array("0914dff4", ndim=1),
    hex_to_array("9ba35411", ndim=1),
    hex_to_array("8e6925af", ndim=1),
    hex_to_array("a51a8b5f", ndim=1),
    hex_to_array("2067fcde", ndim=1),
    hex_to_array("a8b09c1a", ndim=1),
    hex_to_array("93d194cd", ndim=1),
    hex_to_array("be49846e", ndim=1),
    hex_to_array("b75d5b9a", ndim=1),
    hex_to_array("d59aecb8", ndim=1),
    hex_to_array("5bf3c917", ndim=1),
    hex_to_array("fee94248", ndim=1),
    hex_to_array("de8ebe96", ndim=1),
    hex_to_array("b5a9328a", ndim=1),
    hex_to_array("2678a647", ndim=1),
    hex_to_array("98312229", ndim=1),
    hex_to_array("2f6c79b3", ndim=1),
    hex_to_array("812c81ad", ndim=1),
    hex_to_array("dadf48ba", ndim=1),
    hex_to_array("24360af2", ndim=1),
    hex_to_array("fab8b464", ndim=1),
    hex_to_array("98c5bfc9", ndim=1),
    hex_to_array("bebd198e", ndim=1),
    hex_to_array("268c3ba7", ndim=1),
    hex_to_array("09e04214", ndim=1),
    hex_to_array("68007bac", ndim=1),
    hex_to_array("b2df3316", ndim=1),
    hex_to_array("96e939e4", ndim=1),
    hex_to_array("6c518d80", ndim=1),
    hex_to_array("c814e204", ndim=1),
    hex_to_array("76a9fb8a", ndim=1),
    hex_to_array("5025c02d", ndim=1),
    hex_to_array("59c58239", ndim=1),
    hex_to_array("de136967", ndim=1),
    hex_to_array("6ccc5a71", ndim=1),
    hex_to_array("fa256395", ndim=1),
    hex_to_array("9674ee15", ndim=1),
    hex_to_array("5886ca5d", ndim=1),
    hex_to_array("2e2f31d7", ndim=1),
    hex_to_array("7e0af1fa", ndim=1),
    hex_to_array("27cf73c3", ndim=1),
    hex_to_array("749c47ab", ndim=1),
    hex_to_array("18501dda", ndim=1),
    hex_to_array("e2757e4f", ndim=1),
    hex_to_array("7401905a", ndim=1),
    hex_to_array("cafaaae3", ndim=1),
    hex_to_array("e4d59b34", ndim=1),
    hex_to_array("9adf6ace", ndim=1),
    hex_to_array("bd10190d", ndim=1),
    hex_to_array("fe4890d1", ndim=1),
    hex_to_array("e6188d0b", ndim=1),
    hex_to_array("046df344", ndim=1),
    hex_to_array("706c631e", ndim=1),
]).T


@pytest.mark.parametrize("key,out", [
    (key128, w128_true),
    (key192, w192_true),
    (key256, w256_true),
])
def test_expand_key(key, out):
    assert array_equal(expand_key(key), out)


# Test of intermediate values from xor with RCON lookup
asw_to_axo_128 = (
    (hex_to_array("8a84eb01", 1), hex_to_array("8b84eb01", 1)),
    (hex_to_array("50386be5", 1), hex_to_array("52386be5", 1)),
    (hex_to_array("cb42d28f", 1), hex_to_array("cf42d28f", 1)),
    (hex_to_array("dac4e23c", 1), hex_to_array("d2c4e23c", 1)),
    (hex_to_array("2b9563b9", 1), hex_to_array("3b9563b9", 1)),
    (hex_to_array("99596582", 1), hex_to_array("b9596582", 1)),
    (hex_to_array("63dc5474", 1), hex_to_array("23dc5474", 1)),
    (hex_to_array("2486842f", 1), hex_to_array("a486842f", 1)),
    (hex_to_array("5da515d2", 1), hex_to_array("46a515d2", 1)),
    (hex_to_array("4a639f5b", 1), hex_to_array("7c639f5b", 1)),
)
asw_to_axo_192 = (
    (hex_to_array("717f2100", 1), hex_to_array("707f2100", 1)),
    (hex_to_array("b1bb254a", 1), hex_to_array("b3bb254a", 1)),
    (hex_to_array("01ed44ea", 1), hex_to_array("05ed44ea", 1)),
    (hex_to_array("e2048e82", 1), hex_to_array("ea048e82", 1)),
    (hex_to_array("5e49f83e", 1), hex_to_array("4e49f83e", 1)),
    (hex_to_array("8ca96dc3", 1), hex_to_array("aca96dc3", 1)),
    (hex_to_array("82a19e22", 1), hex_to_array("c2a19e22", 1)),
    (hex_to_array("eb94d565", 1), hex_to_array("6b94d565", 1)),
)
asw_to_axo_256 = (
    (hex_to_array("fa9ebf01", 1), hex_to_array("fb9ebf01", 1)),
    (hex_to_array("4c39b8a9", 1), hex_to_array("4e39b8a9", 1)),
    (hex_to_array("50b66d15", 1), hex_to_array("54b66d15", 1)),
    (hex_to_array("e12cfa01", 1), hex_to_array("e92cfa01", 1)),
    (hex_to_array("a61312cb", 1), hex_to_array("b61312cb", 1)),
    (hex_to_array("8a8f2ecc", 1), hex_to_array("aa8f2ecc", 1)),
    (hex_to_array("cad4d77a", 1), hex_to_array("8ad4d77a", 1)),
)


@pytest.mark.parametrize(
    "asw_to_axo",
    [asw_to_axo_128, asw_to_axo_192, asw_to_axo_256]
)
def test_rcon_lookup(asw_to_axo):
    """Test of intermediate values from xor with RCON lookup."""
    for i, (asw, axo) in enumerate(asw_to_axo, 1):
        assert array_equal(xor(asw, RCON[i]), axo), i


# ---------------------------------------------------------------------
# FIPS197 Appendix B - Cipher Example

# Note: these are copied 'in reverse' from the tables, hence why we use
# `.swapaxes()` redundaantly here.  (It's called in `hex_to_array()`)
rounds = (
    (  # 1
        hex_to_array("19 a0 9a e9 3d f4 c6 f8 e3 e2 8d 48 be 2b 2a 08").swapaxes(0, 1),  # Start of round
        hex_to_array("d4 e0 b8 1e 27 bf b4 41 11 98 5d 52 ae f1 e5 30").swapaxes(0, 1),  # After SubBytes
        hex_to_array("d4 e0 b8 1e bf b4 41 27 5d 52 11 98 30 ae f1 e5").swapaxes(0, 1),  # After ShiftRows
        hex_to_array("04 e0 48 28 66 cb f8 06 81 19 d3 26 e5 9a 7a 4c").swapaxes(0, 1),  # After MixColumns
        hex_to_array("a0 88 23 2a fa 54 a3 6c fe 2c 39 76 17 b1 39 05").swapaxes(0, 1),  # Round Key Value
    ),
    (  # 2
        hex_to_array("a4 68 6b 02 9c 9f 5b 6a 7f 35 ea 50 f2 2b 43 49").swapaxes(0, 1),  # Start of round
        hex_to_array("49 45 7f 77 de db 39 02 d2 96 87 53 89 f1 1a 3b").swapaxes(0, 1),  # After SubBytes
        hex_to_array("49 45 7f 77 db 39 02 de 87 53 d2 96 3b 89 f1 1a").swapaxes(0, 1),  # After ShiftRows
        hex_to_array("58 1b db 1b 4d 4b e7 6b ca 5a ca b0 f1 ac a8 e5").swapaxes(0, 1),  # After MixColumns
        hex_to_array("f2 7a 59 73 c2 96 35 59 95 b9 80 f6 f2 43 7a 7f").swapaxes(0, 1),  # Round Key Value
    ),
    (  # 3
        hex_to_array("aa 61 82 68 8f dd d2 32 5f e3 4a 46 03 ef d2 9a").swapaxes(0, 1),  # Start of round
        hex_to_array("ac ef 13 45 73 c1 b5 23 cf 11 d6 5a 7b df b5 b8").swapaxes(0, 1),  # After SubBytes
        hex_to_array("ac ef 13 45 c1 b5 23 73 d6 5a cf 11 b8 7b df b5").swapaxes(0, 1),  # After ShiftRows
        hex_to_array("75 20 53 bb ec 0b c0 25 09 63 cf d0 93 33 7c dc").swapaxes(0, 1),  # After MixColumns
        hex_to_array("3d 47 1e 6d 80 16 23 7a 47 fe 7e 88 7d 3e 44 3b").swapaxes(0, 1),  # Round Key Value
    ),
    (  # 4
        hex_to_array("48 67 4d d6 6c 1d e3 5f 4e 9d b1 58 ee 0d 38 e7").swapaxes(0, 1),  # Start of round
        hex_to_array("52 85 e3 f6 50 a4 11 cf 2f 5e c8 6a 28 d7 07 94").swapaxes(0, 1),  # After SubBytes
        hex_to_array("52 85 e3 f6 a4 11 cf 50 c8 6a 2f 5e 94 28 d7 07").swapaxes(0, 1),  # After ShiftRows
        hex_to_array("0f 60 6f 5e d6 31 c0 b3 da 38 10 13 a9 bf 6b 01").swapaxes(0, 1),  # After MixColumns
        hex_to_array("ef a8 b6 db 44 52 71 0b a5 5b 25 ad 41 7f 3b 00").swapaxes(0, 1),  # Round Key Value
    ),
    (  # 5
        hex_to_array("e0 c8 d9 85 92 63 b1 b8 7f 63 35 be e8 c0 50 01").swapaxes(0, 1),  # Start of round
        hex_to_array("e1 e8 35 97 4f fb c8 6c d2 fb 96 ae 9b ba 53 7c").swapaxes(0, 1),  # After SubBytes
        hex_to_array("e1 e8 35 97 fb c8 6c 4f 96 ae d2 fb 7c 9b ba 53").swapaxes(0, 1),  # After ShiftRows
        hex_to_array("25 bd b6 4c d1 11 3a 4c a9 d1 33 c0 ad 68 8e b0").swapaxes(0, 1),  # After MixColumns
        hex_to_array("d4 7c ca 11 d1 83 f2 f9 c6 9d b8 15 f8 87 bc bc").swapaxes(0, 1),  # Round Key Value
    ),
    (  # 6
        hex_to_array("f1 c1 7c 5d 00 92 c8 b5 6f 4c 8b d5 55 ef 32 0c").swapaxes(0, 1),  # Start of round
        hex_to_array("a1 78 10 4c 63 4f e8 d5 a8 29 3d 03 fc df 23 fe").swapaxes(0, 1),  # After SubBytes
        hex_to_array("a1 78 10 4c 4f e8 d5 63 3d 03 a8 29 fe fc df 23").swapaxes(0, 1),  # After ShiftRows
        hex_to_array("4b 2c 33 37 86 4a 9d d2 8d 89 f4 18 6d 80 e8 d8").swapaxes(0, 1),  # After MixColumns
        hex_to_array("6d 11 db ca 88 0b f9 00 a3 3e 86 93 7a fd 41 fd").swapaxes(0, 1),  # Round Key Value
    ),
    (  # 7
        hex_to_array("26 3d e8 fd 0e 41 64 d2 2e b7 72 8b 17 7d a9 25").swapaxes(0, 1),  # Start of roundz
        hex_to_array("f7 27 9b 54 ab 83 43 b5 31 a9 40 3d f0 ff d3 3f").swapaxes(0, 1),  # After SubBytesz
        hex_to_array("f7 27 9b 54 83 43 b5 ab 40 3d 31 a9 3f f0 ff d3").swapaxes(0, 1),  # After ShiftRowsz
        hex_to_array("14 46 27 34 15 16 46 2a b5 15 56 d8 bf ec d7 43").swapaxes(0, 1),  # After MixColumnsz
        hex_to_array("4e 5f 84 4e 54 5f a6 a6 f7 c9 4f dc 0e f3 b2 4f").swapaxes(0, 1),  # Round Key Valuez
    ),
    (  # 8
        hex_to_array("5a 19 a3 7a 41 49 e0 8c 42 dc 19 04 b1 1f 65 0c").swapaxes(0, 1),  # Start of round
        hex_to_array("be d4 0a da 83 3b e1 64 2c 86 d4 f2 c8 c0 4d fe").swapaxes(0, 1),  # After SubBytes
        hex_to_array("be d4 0a da 3b e1 64 83 d4 f2 2c 86 fe c8 c0 4d").swapaxes(0, 1),  # After ShiftRows
        hex_to_array("00 b1 54 fa 51 c8 76 1b 2f 89 6d 99 d1 ff cd ea").swapaxes(0, 1),  # After MixColumns
        hex_to_array("ea b5 31 7f d2 8d 2b 8d 73 ba f5 29 21 d2 60 2f").swapaxes(0, 1),  # Round Key Value
    ),
    (  # 9
        hex_to_array("ea 04 65 85 83 45 5d 96 5c 33 98 b0 f0 2d ad c5").swapaxes(0, 1),  # Start of round
        hex_to_array("87 f2 4d 97 ec 6e 4c 90 4a c3 46 e7 8c d8 95 a6").swapaxes(0, 1),  # After SubBytes
        hex_to_array("87 f2 4d 97 6e 4c 90 ec 46 e7 4a c3 a6 8c d8 95").swapaxes(0, 1),  # After ShiftRows
        hex_to_array("47 40 a3 4c 37 d4 70 9f 94 e4 3a 42 ed a5 a6 bc").swapaxes(0, 1),  # After MixColumns
        hex_to_array("ac 19 28 57 77 fa d1 5c 66 dc 29 00 f3 21 41 6e").swapaxes(0, 1),  # Round Key Value
    ),
)


@pytest.mark.parametrize("rounds", [rounds])
def test_expand_key_rounds(rounds):
    for i, (start_of_round, after_sub_bytes, after_shift_rows, after_mix_columns, roundkey) in enumerate(rounds):
        assert array_equal(sub_bytes(start_of_round), after_sub_bytes), "bad sub_bytes() at %d" % i
        assert array_equal(shift_rows(after_sub_bytes), after_shift_rows), "bad shift_rows() at %d" % i
        assert array_equal(mix_columns(after_shift_rows), after_mix_columns), "bad mix_columns() at %d" % i


def test_encrypt_small():
    # Test of just start+key -> target.  This is done more comprehensively
    # in Appendix C tests below as well
    start = hex_to_array("32 43 f6 a8 88 5a 30 8d 31 31 98 a2 e0 37 07 34")
    key = hex_to_array("2b 7e 15 16 28 ae d2 a6 ab f7 15 88 09 cf 4f 3c")
    # Column here is copied so is column-ordered
    tgt = array(
        [
            [0x39, 0x02, 0xdc, 0x19],
            [0x25, 0xdc, 0x11, 0x6a],
            [0x84, 0x09, 0x85, 0x0b],
            [0x1d, 0xfb, 0x97, 0x32],
        ], dtype=np.uint8)
    assert array_equal(encrypt_raw(start, key), tgt)


# ---------------------------------------------------------------------
# FIPS197 Appendix C - Example Vectors

# AES-128
aes128_vectors = (
    "00112233445566778899aabbccddeeff",  # PLAINTEXT
    "000102030405060708090a0b0c0d0e0f",  # KEY
    "00112233445566778899aabbccddeeff",  # round[ 0].input
    "000102030405060708090a0b0c0d0e0f",  # round[ 0].k_sch
    "00102030405060708090a0b0c0d0e0f0",  # round[ 1].start
    "63cab7040953d051cd60e0e7ba70e18c",  # round[ 1].s_box
    "6353e08c0960e104cd70b751bacad0e7",  # round[ 1].s_row
    "5f72641557f5bc92f7be3b291db9f91a",  # round[ 1].m_col
    "d6aa74fdd2af72fadaa678f1d6ab76fe",  # round[ 1].k_sch
    "89d810e8855ace682d1843d8cb128fe4",  # round[ 2].start
    "a761ca9b97be8b45d8ad1a611fc97369",  # round[ 2].s_box
    "a7be1a6997ad739bd8c9ca451f618b61",  # round[ 2].s_row
    "ff87968431d86a51645151fa773ad009",  # round[ 2].m_col
    "b692cf0b643dbdf1be9bc5006830b3fe",  # round[ 2].k_sch
    "4915598f55e5d7a0daca94fa1f0a63f7",  # round[ 3].start
    "3b59cb73fcd90ee05774222dc067fb68",  # round[ 3].s_box
    "3bd92268fc74fb735767cbe0c0590e2d",  # round[ 3].s_row
    "4c9c1e66f771f0762c3f868e534df256",  # round[ 3].m_col
    "b6ff744ed2c2c9bf6c590cbf0469bf41",  # round[ 3].k_sch
    "fa636a2825b339c940668a3157244d17",  # round[ 4].start
    "2dfb02343f6d12dd09337ec75b36e3f0",  # round[ 4].s_box
    "2d6d7ef03f33e334093602dd5bfb12c7",  # round[ 4].s_row
    "6385b79ffc538df997be478e7547d691",  # round[ 4].m_col
    "47f7f7bc95353e03f96c32bcfd058dfd",  # round[ 4].k_sch
    "247240236966b3fa6ed2753288425b6c",  # round[ 5].start
    "36400926f9336d2d9fb59d23c42c3950",  # round[ 5].s_box
    "36339d50f9b539269f2c092dc4406d23",  # round[ 5].s_row
    "f4bcd45432e554d075f1d6c51dd03b3c",  # round[ 5].m_col
    "3caaa3e8a99f9deb50f3af57adf622aa",  # round[ 5].k_sch
    "c81677bc9b7ac93b25027992b0261996",  # round[ 6].start
    "e847f56514dadde23f77b64fe7f7d490",  # round[ 6].s_box
    "e8dab6901477d4653ff7f5e2e747dd4f",  # round[ 6].s_row
    "9816ee7400f87f556b2c049c8e5ad036",  # round[ 6].m_col
    "5e390f7df7a69296a7553dc10aa31f6b",  # round[ 6].k_sch
    "c62fe109f75eedc3cc79395d84f9cf5d",  # round[ 7].start
    "b415f8016858552e4bb6124c5f998a4c",  # round[ 7].s_box
    "b458124c68b68a014b99f82e5f15554c",  # round[ 7].s_row
    "c57e1c159a9bd286f05f4be098c63439",  # round[ 7].m_col
    "14f9701ae35fe28c440adf4d4ea9c026",  # round[ 7].k_sch
    "d1876c0f79c4300ab45594add66ff41f",  # round[ 8].start
    "3e175076b61c04678dfc2295f6a8bfc0",  # round[ 8].s_box
    "3e1c22c0b6fcbf768da85067f6170495",  # round[ 8].s_row
    "baa03de7a1f9b56ed5512cba5f414d23",  # round[ 8].m_col
    "47438735a41c65b9e016baf4aebf7ad2",  # round[ 8].k_sch
    "fde3bad205e5d0d73547964ef1fe37f1",  # round[ 9].start
    "5411f4b56bd9700e96a0902fa1bb9aa1",  # round[ 9].s_box
    "54d990a16ba09ab596bbf40ea111702f",  # round[ 9].s_row
    "e9f74eec023020f61bf2ccf2353c21c7",  # round[ 9].m_col
    "549932d1f08557681093ed9cbe2c974e",  # round[ 9].k_sch
    "bd6e7c3df2b5779e0b61216e8b10b689",  # round[10].start
    "7a9f102789d5f50b2beffd9f3dca4ea7",  # round[10].s_box
    "7ad5fda789ef4e272bca100b3d9ff59f",  # round[10].s_row
    "13111d7fe3944a17f307a78b4d2b30c5",  # round[10].k_sch
    "69c4e0d86a7b0430d8cdb78070b4c55a",  # round[10].output
)

# AES-192
aes192_vectors = (
    "00112233445566778899aabbccddeeff",  # PLAINTEXT
    "000102030405060708090a0b0c0d0e0f1011121314151617",  # KEY
    "00112233445566778899aabbccddeeff",  # round[ 0].input
    "000102030405060708090a0b0c0d0e0f",  # round[ 0].k_sch
    "00102030405060708090a0b0c0d0e0f0",  # round[ 1].start
    "63cab7040953d051cd60e0e7ba70e18c",  # round[ 1].s_box
    "6353e08c0960e104cd70b751bacad0e7",  # round[ 1].s_row
    "5f72641557f5bc92f7be3b291db9f91a",  # round[ 1].m_col
    "10111213141516175846f2f95c43f4fe",  # round[ 1].k_sch
    "4f63760643e0aa85aff8c9d041fa0de4",  # round[ 2].start
    "84fb386f1ae1ac977941dd70832dd769",  # round[ 2].s_box
    "84e1dd691a41d76f792d389783fbac70",  # round[ 2].s_row
    "9f487f794f955f662afc86abd7f1ab29",  # round[ 2].m_col
    "544afef55847f0fa4856e2e95c43f4fe",  # round[ 2].k_sch
    "cb02818c17d2af9c62aa64428bb25fd7",  # round[ 3].start
    "1f770c64f0b579deaaac432c3d37cf0e",  # round[ 3].s_box
    "1fb5430ef0accf64aa370cde3d77792c",  # round[ 3].s_row
    "b7a53ecbbf9d75a0c40efc79b674cc11",  # round[ 3].m_col
    "40f949b31cbabd4d48f043b810b7b342",  # round[ 3].k_sch
    "f75c7778a327c8ed8cfebfc1a6c37f53",  # round[ 4].start
    "684af5bc0acce85564bb0878242ed2ed",  # round[ 4].s_box
    "68cc08ed0abbd2bc642ef555244ae878",  # round[ 4].s_row
    "7a1e98bdacb6d1141a6944dd06eb2d3e",  # round[ 4].m_col
    "58e151ab04a2a5557effb5416245080c",  # round[ 4].k_sch
    "22ffc916a81474416496f19c64ae2532",  # round[ 5].start
    "9316dd47c2fa92834390a1de43e43f23",  # round[ 5].s_box
    "93faa123c2903f4743e4dd83431692de",  # round[ 5].s_row
    "aaa755b34cffe57cef6f98e1f01c13e6",  # round[ 5].m_col
    "2ab54bb43a02f8f662e3a95d66410c08",  # round[ 5].k_sch
    "80121e0776fd1d8a8d8c31bc965d1fee",  # round[ 6].start
    "cdc972c53854a47e5d64c765904cc028",  # round[ 6].s_box
    "cd54c7283864c0c55d4c727e90c9a465",  # round[ 6].s_row
    "921f748fd96e937d622d7725ba8ba50c",  # round[ 6].m_col
    "f501857297448d7ebdf1c6ca87f33e3c",  # round[ 6].k_sch
    "671ef1fd4e2a1e03dfdcb1ef3d789b30",  # round[ 7].start
    "8572a1542fe5727b9e86c8df27bc1404",  # round[ 7].s_box
    "85e5c8042f8614549ebca17b277272df",  # round[ 7].s_row
    "e913e7b18f507d4b227ef652758acbcc",  # round[ 7].m_col
    "e510976183519b6934157c9ea351f1e0",  # round[ 7].k_sch
    "0c0370d00c01e622166b8accd6db3a2c",  # round[ 8].start
    "fe7b5170fe7c8e93477f7e4bf6b98071",  # round[ 8].s_box
    "fe7c7e71fe7f807047b95193f67b8e4b",  # round[ 8].s_row
    "6cf5edf996eb0a069c4ef21cbfc25762",  # round[ 8].m_col
    "1ea0372a995309167c439e77ff12051e",  # round[ 8].k_sch
    "7255dad30fb80310e00d6c6b40d0527c",  # round[ 9].start
    "40fc5766766c7bcae1d7507f09700010",  # round[ 9].s_box
    "406c501076d70066e17057ca09fc7b7f",  # round[ 9].s_row
    "7478bcdce8a50b81d4327a9009188262",  # round[ 9].m_col
    "dd7e0e887e2fff68608fc842f9dcc154",  # round[ 9].k_sch
    "a906b254968af4e9b4bdb2d2f0c44336",  # round[10].start
    "d36f3720907ebf1e8d7a37b58c1c1a05",  # round[10].s_box
    "d37e3705907a1a208d1c371e8c6fbfb5",  # round[10].s_row
    "0d73cc2d8f6abe8b0cf2dd9bb83d422e",  # round[10].m_col
    "859f5f237a8d5a3dc0c02952beefd63a",  # round[10].k_sch
    "88ec930ef5e7e4b6cc32f4c906d29414",  # round[11].start
    "c4cedcabe694694e4b23bfdd6fb522fa",  # round[11].s_box
    "c494bffae62322ab4bb5dc4e6fce69dd",  # round[11].s_row
    "71d720933b6d677dc00b8f28238e0fb7",  # round[11].m_col
    "de601e7827bcdf2ca223800fd8aeda32",  # round[11].k_sch
    "afb73eeb1cd1b85162280f27fb20d585",  # round[12].start
    "79a9b2e99c3e6cd1aa3476cc0fb70397",  # round[12].s_box
    "793e76979c3403e9aab7b2d10fa96ccc",  # round[12].s_row
    "a4970a331a78dc09c418c271e3a41d5d",  # round[12].k_sch
    "dda97ca4864cdfe06eaf70a0ec0d7191",  # round[12].output
)

# AES-256
aes256_vectors = (
    "00112233445566778899aabbccddeeff",  # PLAINTEXT
    "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",  # KEY
    "00112233445566778899aabbccddeeff",  # round[ 0].input
    "000102030405060708090a0b0c0d0e0f",  # round[ 0].k_sch
    "00102030405060708090a0b0c0d0e0f0",  # round[ 1].start
    "63cab7040953d051cd60e0e7ba70e18c",  # round[ 1].s_box
    "6353e08c0960e104cd70b751bacad0e7",  # round[ 1].s_row
    "5f72641557f5bc92f7be3b291db9f91a",  # round[ 1].m_col
    "101112131415161718191a1b1c1d1e1f",  # round[ 1].k_sch
    "4f63760643e0aa85efa7213201a4e705",  # round[ 2].start
    "84fb386f1ae1ac97df5cfd237c49946b",  # round[ 2].s_box
    "84e1fd6b1a5c946fdf4938977cfbac23",  # round[ 2].s_row
    "bd2a395d2b6ac438d192443e615da195",  # round[ 2].m_col
    "a573c29fa176c498a97fce93a572c09c",  # round[ 2].k_sch
    "1859fbc28a1c00a078ed8aadc42f6109",  # round[ 3].start
    "adcb0f257e9c63e0bc557e951c15ef01",  # round[ 3].s_box
    "ad9c7e017e55ef25bc150fe01ccb6395",  # round[ 3].s_row
    "810dce0cc9db8172b3678c1e88a1b5bd",  # round[ 3].m_col
    "1651a8cd0244beda1a5da4c10640bade",  # round[ 3].k_sch
    "975c66c1cb9f3fa8a93a28df8ee10f63",  # round[ 4].start
    "884a33781fdb75c2d380349e19f876fb",  # round[ 4].s_box
    "88db34fb1f807678d3f833c2194a759e",  # round[ 4].s_row
    "b2822d81abe6fb275faf103a078c0033",  # round[ 4].m_col
    "ae87dff00ff11b68a68ed5fb03fc1567",  # round[ 4].k_sch
    "1c05f271a417e04ff921c5c104701554",  # round[ 5].start
    "9c6b89a349f0e18499fda678f2515920",  # round[ 5].s_box
    "9cf0a62049fd59a399518984f26be178",  # round[ 5].s_row
    "aeb65ba974e0f822d73f567bdb64c877",  # round[ 5].m_col
    "6de1f1486fa54f9275f8eb5373b8518d",  # round[ 5].k_sch
    "c357aae11b45b7b0a2c7bd28a8dc99fa",  # round[ 6].start
    "2e5bacf8af6ea9e73ac67a34c286ee2d",  # round[ 6].s_box
    "2e6e7a2dafc6eef83a86ace7c25ba934",  # round[ 6].s_row
    "b951c33c02e9bd29ae25cdb1efa08cc7",  # round[ 6].m_col
    "c656827fc9a799176f294cec6cd5598b",  # round[ 6].k_sch
    "7f074143cb4e243ec10c815d8375d54c",  # round[ 7].start
    "d2c5831a1f2f36b278fe0c4cec9d0329",  # round[ 7].s_box
    "d22f0c291ffe031a789d83b2ecc5364c",  # round[ 7].s_row
    "ebb19e1c3ee7c9e87d7535e9ed6b9144",  # round[ 7].m_col
    "3de23a75524775e727bf9eb45407cf39",  # round[ 7].k_sch
    "d653a4696ca0bc0f5acaab5db96c5e7d",  # round[ 8].start
    "f6ed49f950e06576be74624c565058ff",  # round[ 8].s_box
    "f6e062ff507458f9be50497656ed654c",  # round[ 8].s_row
    "5174c8669da98435a8b3e62ca974a5ea",  # round[ 8].m_col
    "0bdc905fc27b0948ad5245a4c1871c2f",  # round[ 8].k_sch
    "5aa858395fd28d7d05e1a38868f3b9c5",  # round[ 9].start
    "bec26a12cfb55dff6bf80ac4450d56a6",  # round[ 9].s_box
    "beb50aa6cff856126b0d6aff45c25dc4",  # round[ 9].s_row
    "0f77ee31d2ccadc05430a83f4ef96ac3",  # round[ 9].m_col
    "45f5a66017b2d387300d4d33640a820a",  # round[ 9].k_sch
    "4a824851c57e7e47643de50c2af3e8c9",  # round[10].start
    "d61352d1a6f3f3a04327d9fee50d9bdd",  # round[10].s_box
    "d6f3d9dda6279bd1430d52a0e513f3fe",  # round[10].s_row
    "bd86f0ea748fc4f4630f11c1e9331233",  # round[10].m_col
    "7ccff71cbeb4fe5413e6bbf0d261a7df",  # round[10].k_sch
    "c14907f6ca3b3aa070e9aa313b52b5ec",  # round[11].start
    "783bc54274e280e0511eacc7e200d5ce",  # round[11].s_box
    "78e2acce741ed5425100c5e0e23b80c7",  # round[11].s_row
    "af8690415d6e1dd387e5fbedd5c89013",  # round[11].m_col
    "f01afafee7a82979d7a5644ab3afe640",  # round[11].k_sch
    "5f9c6abfbac634aa50409fa766677653",  # round[12].start
    "cfde0208f4b418ac5309db5c338538ed",  # round[12].s_box
    "cfb4dbedf4093808538502ac33de185c",  # round[12].s_row
    "7427fae4d8a695269ce83d315be0392b",  # round[12].m_col
    "2541fe719bf500258813bbd55a721c0a",  # round[12].k_sch
    "516604954353950314fb86e401922521",  # round[13].start
    "d133f22a1aed2a7bfa0f44697c4f3ffd",  # round[13].s_box
    "d1ed44fd1a0f3f2afa4ff27b7c332a69",  # round[13].s_row
    "2c21a820306f154ab712c75eee0da04f",  # round[13].m_col
    "4e5a6699a9f24fe07e572baacdf8cdea",  # round[13].k_sch
    "627bceb9999d5aaac945ecf423f56da5",  # round[14].start
    "aa218b56ee5ebeacdd6ecebf26e63c06",  # round[14].s_box
    "aa5ece06ee6e3c56dde68bac2621bebf",  # round[14].s_row
    "24fc79ccbf0979e9371ac23c6d68de36",  # round[14].k_sch
    "8ea2b7ca516745bfeafc49904b496089",  # round[14].output
)


@pytest.mark.parametrize(
    "vectors",
    [aes128_vectors, aes192_vectors, aes256_vectors]
)
def test_example_vectors(vectors):
    # It's assumed that all of these are given in 1d form, hence we
    # only need the single call to `.swapaxes(0, 1)` that occur
    # already in `hex_to_array()`
    start, key, tgt = map(
        hex_to_array,
        (vectors[0], vectors[1], vectors[-1])
    )
    assert array_equal(encrypt_raw(start, key), tgt)
