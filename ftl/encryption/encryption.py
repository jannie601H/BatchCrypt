import datetime
import tensorflow as tf
import numpy as np
from numba import njit, prange
import math
import random
from ftl.encryption.paillier import PaillierPublicKey, PaillierPrivateKey
from ftl.encryption import aciq

import multiprocessing
from joblib import Parallel, delayed

import warnings
import pysnooper

N_JOBS = multiprocessing.cpu_count()


def encrypt(public_key: PaillierPublicKey, x):
    return public_key.encrypt(x)


def encrypt_array(public_key: PaillierPublicKey, A):
    # encrypt_A = []
    # for i in range(len(A)):
    #     encrypt_A.append(public_key.encrypt(float(A[i])))
    encrypt_A = Parallel(n_jobs=N_JOBS)(delayed(public_key.encrypt)(num) for num in A)
    return np.array(encrypt_A)


def encrypt_matrix(public_key: PaillierPublicKey, A):
    og_shape = A.shape
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)

    # print('encrypting matrix shaped ' + str(og_shape))
    A = np.reshape(A, (1, -1))
    A = np.squeeze(A)
    # print('max = ' + str(np.amax(A)))
    # print('min = ' + str(np.amin(A)))
    # encrypt_A = []
    # for i in range(len(A)):
    #     row = []
    #     for j in range(len(A[i])):
    #         if len(A.shape) == 3:
    #             row.append([public_key.encrypt(float(A[i, j, k])) for k in range(len(A[i][j]))])
    #         else:
    #             row.append(public_key.encrypt(float(A[i, j])))
    #
    #     encrypt_A.append(row)
    encrypt_A = Parallel(n_jobs=N_JOBS)(delayed(public_key.encrypt)(num) for num in A)
    encrypt_A = np.expand_dims(encrypt_A, axis=0)
    encrypt_A = np.reshape(encrypt_A, og_shape)
    return np.array(encrypt_A)


@njit(parallel=True)
def stochastic_r(ori, frac, rand):
    result = np.zeros(len(ori), dtype=np.int32)
    for i in prange(len(ori)):
        if frac[i] >= 0:
            result[i] = np.floor(ori[i]) if frac[i] <= rand[i] else np.ceil(ori[i])
        else:
            result[i] = np.floor(ori[i]) if (-1 * frac[i]) > rand[i] else np.ceil(ori[i])
    return result


def stochastic_round(ori):
    rand = np.random.rand(len(ori))
    frac, decim = np.modf(ori)
    # result = np.zeros(len(ori))
    # for i in range(len(ori)):
    #     if frac[i] >= 0:
    #         result[i] = np.floor(ori[i]) if frac[i] <= rand[i] else np.ceil(ori[i])
    #     else:
    #         result[i] = np.floor(ori[i]) if (-1 * frac[i]) > rand[i] else np.ceil(ori[i])
    result = stochastic_r(ori, frac, rand)
    return result.astype(np.int32)


def stochastic_round_matrix(ori):
    _shape = ori.shape
    ori = np.reshape(ori, (1, -1))
    ori = np.squeeze(ori)
    # rand = np.random.rand(len(ori))
    # frac, decim = np.modf(ori)

    # result = np.zeros(len(ori))
    #
    # for i in range(len(ori)):
    #     if frac[i] >= 0:
    #         result[i] = np.floor(ori[i]) if frac[i] <= rand[i] else np.ceil(ori[i])
    #     else:
    #         result[i] = np.floor(ori[i]) if (-1 * frac[i]) > rand[i] else np.ceil(ori[i])
    result = stochastic_round(ori)
    result = result.reshape(_shape)
    return result


def quantize_matrix(matrix, bit_width=8, r_max=0.5):
    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = (uns_matrix * (pow(2, bit_width - 1) - 1.0) / r_max)
    result = (og_sign * uns_result)
    # result = np.reshape(result, (1, -1))
    # result = np.squeeze(result)
    # # print(result)
    # result = stochastic_round(result)
    # print(result)
    return result, og_sign


def quantize_matrix_stochastic(matrix, bit_width=8, r_max=0.5):
    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = (uns_matrix * (pow(2, bit_width - 1) - 1.0) / r_max)
    result = (og_sign * uns_result)
    # result = np.reshape(result, (1, -1))
    # result = np.squeeze(result)
    # # print(result)
    result = stochastic_round_matrix(result)
    # print(result)
    return result, og_sign


def unquantize_matrix(matrix, bit_width=8, r_max=0.5):
    matrix = matrix.astype(int)
    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = uns_matrix * r_max / (pow(2, bit_width - 1) - 1.0)
    result = og_sign * uns_result
    return result.astype(np.float32)


def true_to_two_comp(input, bit_width):
    def true_to_two(value, bit_width):
        if value < 0:
            return 2 ** (bit_width + 1) + value
        else:
            return value

    # two_strings = [np.binary_repr(x, bit_width) for x in input]
    # # use 2 bits for sign
    # result = [int(x[0] + x, 2) for x in two_strings]
    result = Parallel(n_jobs=N_JOBS)(delayed(true_to_two)(x, bit_width) for x in input)
    return np.array(result)


# @njit(parallel=True)
# def true_to_two_comp_(input, bit_width=8, pad_zero=3):
#     result = np.zeros(len(input), dtype=np.int32)
#     for i in prange(len(input)):
#         if input[i] >= 0:
#             result[i] = input[i]
#         else:
#             result[i] = 2 ** (bit_width + 1) + input[i]
#     return result

@njit(parallel=True)
def true_to_two_comp_(input, bit_width=8, pad_zero=3):
    total_bits = bit_width + pad_zero
    offset = 2 ** total_bits
    result = np.zeros(len(input), dtype=np.int32)
    for i in prange(len(input)):
        result[i] = input[i] if input[i] >= 0 else input[i] + offset
    return result

# @pysnooper.snoop('en_batch.log')
def encrypt_matrix_batch(public_key: PaillierPublicKey, A, batch_size=16, bit_width=8, pad_zero=3, r_max=0.5):
    og_shape = A.shape
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)

    # print('encrypting matrix shaped ' + str(og_shape) + ' ' + str(datetime.datetime.now().time()))

    A, og_sign = quantize_matrix(A, bit_width, r_max)

    A = np.reshape(A, (1, -1))
    A = np.squeeze(A)
    A = stochastic_round(A)
    # print(f"[DEBUG] Quantize→Round 직후 정수 Q_int: min={np.min(A)}, max={np.max(A)}")


    # print("encrpting # " + str(len(A)) + " shape" + str(og_shape)+' ' + str(datetime.datetime.now().time()))

    A_len = len(A)
    # pad array at the end so tha the array is the size of
    A = A if (A_len % batch_size) == 0 \
        else np.pad(A, (0, batch_size - (A_len % batch_size)), 'constant', constant_values=(0, 0))

    # print('padded ' + str(datetime.datetime.now().time()))

    A = true_to_two_comp_(A, bit_width)

    # print([bin(x) for x in A])
    # print("encrpting padded # " + str(len(A))+' ' + str(datetime.datetime.now().time()))


    idx_range = int(len(A) / batch_size)
    idx_base = list(range(idx_range))
    # batched_nums = np.zeros(idx_range, dtype=int)
    batched_nums = np.array([pow(2, 2048)] * idx_range)
    batched_nums *= 0
    # print(batched_nums.dtype)
    for i in range(batch_size):
        idx_filter = [i + x * batch_size for x in idx_base]
        # print(idx_filter)
        filted_num = A[idx_filter]
        # print([bin(x) for x in filted_num])
        batched_nums = (batched_nums * pow(2, (bit_width + pad_zero))) + filted_num
        # print([bin(x) for x in batched_nums])

    # print("encrpting batched # " + str(len(batched_nums))+' ' + str(datetime.datetime.now().time()))

    # print([bin(x).zfill(batch_size*(bit_width+pad_zero) + 2) + ' ' for x in batched_nums])



    encrypt_A = Parallel(n_jobs=N_JOBS)(delayed(public_key.encrypt)(num) for num in batched_nums)

    # print('encryption done'+' ' + str(datetime.datetime.now().time()))
    return encrypt_A, og_shape

# @pysnooper.snoop('en_batch.log')
def encrypt_matrix_batch_zero(public_key: PaillierPublicKey, A, epoch, batch_size=16, bit_width=8, pad_zero=3, r_max=0.5):
    og_shape = A.shape
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)

    Q, _ = quantize_matrix(A, bit_width, r_max)
    Q = Q.reshape(-1)
    Q = stochastic_round(Q).astype(np.int32)
    # print(f"[DEBUG] Quantize→Round 직후 정수 Q_int: min={np.min(Q)}, max={np.max(Q)}")

    # pad_len = (batch_size - (len(Q) % batch_size)) % batch_size
    # padded = np.pad(Q, (0, pad_len), 'constant', constant_values=0)
    Q_len = len(Q)
    # pad array at the end so tha the array is the size of
    Q = Q if (Q_len % batch_size) == 0 \
        else np.pad(Q, (0, batch_size - (Q_len % batch_size)), 'constant', constant_values=(0, 0))

    # block 단위 skip_mask, 3 epoch 이전에는 skip 비활성화.
    # skip_mask_blocks = (np.abs(Q) > 1e-6).reshape(-1, batch_size).any(axis=1)
    skip_mask_blocks = (Q != 0).reshape(-1, batch_size).any(axis=1)

    # print("skip mask blocks: ", skip_mask_blocks)

    # 2’s-complement
    Q = true_to_two_comp_(Q, bit_width)

    # Pack (bit shift + OR)
    idx_range = int(len(Q) / batch_size)
    idx_base = list(range(idx_range))
    batched_nums = np.array([pow(2, 2048)] * idx_range)
    batched_nums *= 0

    for i in range(batch_size):
        idx_filter = [i + x * batch_size for x in idx_base]
        filted_num = Q[idx_filter]
        batched_nums = (batched_nums * pow(2, (bit_width + pad_zero))) + filted_num

    # Paillier encryption
    # encrypt_A = Parallel(n_jobs=N_JOBS)(delayed(public_key.encrypt)(num) for num in batched_nums)
    # print("start encryption")
    encrypt_A = Parallel(n_jobs=N_JOBS)(
        delayed(public_key.encrypt)(batched_nums[i])
        for i in range(len(batched_nums)) if skip_mask_blocks[i]
    )
    # print("end encryption")

    return encrypt_A, og_shape, skip_mask_blocks


def encrypt_matmul(public_key: PaillierPublicKey, A, encrypted_B):
    """
     matrix multiplication between a plain matrix and an encrypted matrix

    :param public_key:
    :param A:
    :param encrypted_B:
    :return:
    """
    if A.shape[-1] != encrypted_B.shape[0]:
        print("A and encrypted_B shape are not consistent")
        exit(1)
    # TODO: need a efficient way to do this?
    res = [[public_key.encrypt(0) for _ in range(encrypted_B.shape[1])] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(encrypted_B.shape[1]):
            for m in range(len(A[i])):
                res[i][j] += A[i][m] * encrypted_B[m][j]
    return np.array(res)


def encrypt_matmul_3(public_key: PaillierPublicKey, A, encrypted_B):
    if A.shape[0] != encrypted_B.shape[0]:
        print("A and encrypted_B shape are not consistent")
        print(A.shape)
        print(encrypted_B.shape)
        exit(1)
    res = []
    for i in range(len(A)):
        res.append(encrypt_matmul(public_key, A[i], encrypted_B[i]))
    return np.array(res)


def decrypt(private_key: PaillierPrivateKey, x):
    return private_key.decrypt(x)


def decrypt_scalar(private_key: PaillierPrivateKey, x):
    return private_key.decrypt(x)


def decrypt_array(private_key: PaillierPrivateKey, X):
    decrypt_x = []
    for i in range(X.shape[0]):
        elem = private_key.decrypt(X[i])
        decrypt_x.append(elem)
    return decrypt_x


def encrypt_array(private_key: PaillierPrivateKey, X):
    decrpt_X = Parallel(n_jobs=N_JOBS)(delayed(private_key.decrypt())(num) for num in X)
    return np.array(decrpt_X)


def decrypt_matrix(private_key: PaillierPrivateKey, A):
    og_shape = A.shape
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)

    A = np.reshape(A, (1, -1))
    A = np.squeeze(A)

    # decrypt_A = []
    # for i in range(len(A)):
    #     row = []
    #     for j in range(len(A[i])):
    #         if len(A.shape) == 3:
    #             row.append([private_key.decrypt(A[i, j, k]) for k in range(len(A[i][j]))])
    #         else:
    #             row.append(private_key.decrypt(A[i, j]))
    #     decrypt_A.append(row)
    decrypt_A = Parallel(n_jobs=N_JOBS)(delayed(private_key.decrypt)(num) for num in A)

    decrypt_A = np.expand_dims(decrypt_A, axis=0)
    decrypt_A = np.reshape(decrypt_A, og_shape)

    return np.array(decrypt_A)


def two_comp_to_true(two_comp, bit_width=8, pad_zero=3):
    def binToInt(s, _bit_width=8):
        return int(s[1:], 2) - int(s[0]) * (1 << (_bit_width - 1))

    if two_comp < 0:
        raise Exception("Error: not expecting negtive value")
    two_com_string = bin(two_comp)[2:].zfill(bit_width + pad_zero)
    sign = two_com_string[0:pad_zero + 1]
    literal = two_com_string[pad_zero + 1:]

    if sign == '0' * (pad_zero + 1):  # positive value
        value = int(literal, 2)
        return value
    elif sign == '0' * (pad_zero - 2) + '1' + '0' * 2:  # positive value
        value = int(literal, 2)
        return value
    elif sign == '0' * pad_zero + '1':  # positive overflow
        value = pow(2, bit_width - 1) - 1
        return value
    elif sign == '0' * (pad_zero - 1) + '1' * 2:  # negtive value
        # if literal == '0' * (bit_width - 1):
        #     return 0
        return binToInt('1' + literal, bit_width)
    elif sign == '0' * (pad_zero - 2) + '1' * 3:  # negtive value
        # if literal == '0' * (bit_width - 1):
        #     return 0
        return binToInt('1' + literal, bit_width)
    elif sign == '0' * (pad_zero - 2) + '110':  # negtive overflow
        print('neg overflow: ' + two_com_string)
        return - (pow(2, bit_width - 1) - 1)
    else:  # unrecognized overflow
        #print('unrecognized overflow: ' + two_com_string)
        warnings.warn('Overflow detected, consider using longer r_max')
        return - (pow(2, bit_width - 1) - 1)


# def two_comp_to_true_(two_comp, bit_width=8, pad_zero=3):
#     def two_comp_lit_to_ori(lit, _bit_width):  # convert 2's complement coding of neg value to its original form
#         return - 1 * (2 ** (_bit_width - 1) - lit)

#     if two_comp < 0:
#         raise Exception("Error: not expecting negtive value")
#     # two_com_string = bin(two_comp)[2:].zfill(bit_width+pad_zero)

#     sign = two_comp >> (bit_width - 1)
#     literal = two_comp & (2 ** (bit_width - 1) - 1)

#     if sign == 0:  # positive value
#         return literal
#     elif sign == 4:  # positive value, 0100
#         return literal
#     elif sign == 1:  # positive overflow, 0001
#         return pow(2, bit_width - 1) - 1
#     elif sign == 3:  # negtive value, 0011
#         return two_comp_lit_to_ori(literal, bit_width)
#     elif sign == 7:  # negtive value, 0111
#         return two_comp_lit_to_ori(literal, bit_width)
#     elif sign == 6:  # negtive overflow, 0110
#         print('neg overflow: ' + str(two_comp))
#         return - (pow(2, bit_width - 1) - 1)
#     else:  # unrecognized overflow
#         #print('unrecognized overflow: ' + str(two_comp))
#         warnings.warn('Overflow detected, consider using longer r_max')
#         return - (pow(2, bit_width - 1) - 1)

def two_comp_to_true_(val, bit_width=8, pad_zero=3):
    total_bits = bit_width + pad_zero
    limit = 2 ** total_bits
    sign_bit = 2 ** (total_bits - 1)
    if val >= sign_bit:
        return val - limit  # 음수
    else:
        return val  # 양수

def restore_shape(component, shape, batch_size=16, bit_width=8, pad_zero=3):
    num_ele = np.prod(shape)
    num_ele_w_pad = batch_size * len(component)
    # print("restoring shape " + str(shape))
    # print(" num_ele %d, num_ele_w_pad %d" % (num_ele, num_ele_w_pad))

    un_batched_nums = np.zeros(num_ele_w_pad, dtype=int)

    for i in range(batch_size):
        filter_ = (pow(2, bit_width + pad_zero) - 1) << ((bit_width + pad_zero) * i)
        # print(bin(filter))
        # filtered_nums = [x & filter for x in component]
        for j in range(len(component)):
            two_comp = (filter_ & component[j]) >> ((bit_width + pad_zero) * i)
            # print(bin(two_comp))
            un_batched_nums[batch_size * j + batch_size - 1 - i] = two_comp_to_true_(two_comp, bit_width, pad_zero)

    un_batched_nums = un_batched_nums[:num_ele]

    re = np.reshape(un_batched_nums, shape)
    # print("reshaped " + str(re.shape))
    return re

# @pysnooper.snoop('de_batch.log')
def decrypt_matrix_batch(private_key: PaillierPrivateKey, A, og_shape, batch_size=16, bit_width=8,
                         pad_zero=3, r_max=0.5):
    # A = [x.ciphertext(be_secure=False) if x.exponent == 0 else
    #      (x.decrease_exponent_to(0).ciphertext(be_secure=False) if x.exponent > 0 else
    #       x.increase_exponent_to(0).ciphertext(be_secure=False)) for x in A]

    # print("decrypting # " + str(len(A)) + " shape " + str(og_shape))

    decrypt_A = Parallel(n_jobs=N_JOBS)(delayed(private_key.decrypt)(num) for num in A)
    decrypt_A = np.array(decrypt_A)
    # print([bin(x).zfill(batch_size*(bit_width+pad_zero) + 2) for x in decrypt_A])


    result = restore_shape(decrypt_A, og_shape, batch_size, bit_width, pad_zero)


    result = unquantize_matrix(result, bit_width, r_max)
    # print(f"[DEBUG] Unquantize 직후 복원 float: min={np.min(result):.6f}, max={np.max(result):.6f}")

    return result


# @pysnooper.snoop('de_batch.log')
def decrypt_matrix_batch_zero(private_key: PaillierPrivateKey, enc_blocks, og_shape,
                         batch_size=16, bit_width=8, pad_zero=3, r_max=0.5):
    """
    Batch Zero 모드용 복호화 함수.
    - 서버에서 합쳐진 enc_blocks를 decrypt → skip_mask_blocks 기준으로 0 블록 채우기 → unpack → unquantize
    """
    # 1) 암호문 블록 전부 복호화
    # print("start decryption")
    decrypted = Parallel(n_jobs=N_JOBS)(delayed(private_key.decrypt)(blk) for blk in enc_blocks)
    decrypted = np.array(decrypted)
    # print("end decryption")

    # 2) Unpack: 각 big_int를 batch_size개의 two_comp 값으로 분리
    block_count = len(decrypted)
    total_elems = block_count * batch_size
    un_batched = np.zeros(total_elems, dtype=int)
    mask_len = bit_width + pad_zero

    for j in range(block_count):
        big_int = int(decrypted[j])
        for i in range(batch_size):
            shift = (batch_size - 1 - i) * mask_len
            mask_ = ((1 << mask_len) - 1) << shift
            two_comp_val = (big_int & mask_) >> shift
            # signed int 복원
            un_batched[j * batch_size + i] = two_comp_to_true_(two_comp_val, bit_width, pad_zero)

    # 3) Padding 제거
    num_elements = int(np.prod(og_shape))
    un_batched = un_batched[:num_elements]
    # print("[DEBUG] Sample unpacked ints (first 20):", un_batched[:20])

    # 4) Shape 복원
    re = un_batched.reshape(og_shape)

    # 5) unquantize
    result = unquantize_matrix(re, bit_width, r_max)
    # print(f"[DEBUG] Unquantize 직후 복원 float: min={np.min(result):.6f}, max={np.max(result):.6f}")

    return result

def calculate_clip_threshold(grads, theta=2.5):
    return [theta * np.std(x) for x in grads]


def calculate_clip_threshold_sparse(grads, theta=2.5):
    result = []
    for layer in grads:
        if isinstance(layer, tf.IndexedSlices):
            result.append(theta * np.std(layer.values.numpy()))
        else:
            result.append(theta * np.std(layer.numpy()))
    return result


def clip_with_threshold(grads, thresholds):
    return [np.clip(x, -1 * y, y) for x, y in zip(grads, thresholds)]


def clip_gradients_std(grads, std_theta=2.5):
    results = []
    thresholds = []
    for component in grads:
        clip_T = np.std(component) * std_theta
        thresholds.append(clip_T)
        results.append(np.clip(component, -1 * clip_T, clip_T))
    return results, thresholds


# def calculate_clip_threshold_aciq_g(grads, bit_width=8):
#     return [aciq.get_alpha_gaus(x, bit_width) for x in grads]
def calculate_clip_threshold_aciq_g(grads, grads_sizes, bit_width=8):
    print("ACIQ bit width:", bit_width)
    res = []
    for idx in range(len(grads)):
        res.append(aciq.get_alpha_gaus(grads[idx], grads_sizes[idx], bit_width))
    # return [aciq.get_alpha_gaus(x, bit_width) for x in grads]
    return res


def calculate_clip_threshold_aciq_l(grads, bit_width=8):
    return [aciq.get_alpha_laplace(x, bit_width) for x in grads]


    ###################################################################


    
