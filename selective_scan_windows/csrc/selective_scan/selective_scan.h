/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include "windows_compat.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SSMScanParamsBase {
    using index_t = uint32_t;

    int batch, seqlen, n_chunks;
    index_t a_batch_stride;
    index_t b_batch_stride;
    index_t out_batch_stride;

    // Common data pointers.
    void *SSW_RESTRICT a_ptr;
    void *SSW_RESTRICT b_ptr;
    void *SSW_RESTRICT out_ptr;
    void *SSW_RESTRICT x_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SSMParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, dstate, n_groups, n_chunks;
    int dim_ngroups_ratio;

    bool delta_softplus;

    index_t A_d_stride;
    index_t A_dstate_stride;
    index_t B_batch_stride;
    index_t B_d_stride;
    index_t B_dstate_stride;
    index_t B_group_stride;
    index_t C_batch_stride;
    index_t C_d_stride;
    index_t C_dstate_stride;
    index_t C_group_stride;
    index_t u_batch_stride;
    index_t u_d_stride;
    index_t delta_batch_stride;
    index_t delta_d_stride;
    index_t out_batch_stride;
    index_t out_d_stride;

    // Common data pointers.
    void *SSW_RESTRICT A_ptr;
    void *SSW_RESTRICT B_ptr;
    void *SSW_RESTRICT C_ptr;
    void *SSW_RESTRICT D_ptr;
    void *SSW_RESTRICT u_ptr;
    void *SSW_RESTRICT delta_ptr;
    void *SSW_RESTRICT delta_bias_ptr;
    void *SSW_RESTRICT out_ptr;
    void *SSW_RESTRICT x_ptr;
};

struct SSMParamsBwd: public SSMParamsBase {
    index_t dout_batch_stride;
    index_t dout_d_stride;
    index_t dA_d_stride;
    index_t dA_dstate_stride;
    index_t dB_batch_stride;
    index_t dB_group_stride;
    index_t dB_d_stride;
    index_t dB_dstate_stride;
    index_t dC_batch_stride;
    index_t dC_group_stride;
    index_t dC_d_stride;
    index_t dC_dstate_stride;
    index_t du_batch_stride;
    index_t du_d_stride;
    index_t ddelta_batch_stride;
    index_t ddelta_d_stride;

    // Common data pointers.
    void *SSW_RESTRICT dout_ptr;
    void *SSW_RESTRICT dA_ptr;
    void *SSW_RESTRICT dB_ptr;
    void *SSW_RESTRICT dC_ptr;
    void *SSW_RESTRICT dD_ptr;
    void *SSW_RESTRICT du_ptr;
    void *SSW_RESTRICT ddelta_ptr;
    void *SSW_RESTRICT ddelta_bias_ptr;
};
