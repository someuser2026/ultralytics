# Source provenance and third-party notices

This package contains a self-contained copy of the core selective-scan sources.
The original directory is not a build or runtime dependency.

Source repository revision: `7af8f2f9bcd0e92d32d7a693400e5d610dc6216c`.

The surrounding repository is distributed under AGPL-3.0 (see LICENSE).
Original Tri Dao / Albert Gu copyright headers and NVIDIA redistribution
notices are retained in the copied files. The source extension metadata
identifies its upstream implementation as BSD-licensed; this port does not
replace or remove third-party license terms.

## Original file SHA-256 hashes

```json
{
  "selective_scan/csrc/selective_scan/selective_scan.h": "a137e4c785fed3d6f9c6c4315978ff99ec7ef464e76f5ed043d22ede3f0e927f",
  "selective_scan/csrc/selective_scan/selective_scan_common.h": "e4569b7a2994a0687f8d6db4e72b968d6c164a5f65da7c568a596affa98bcbf6",
  "selective_scan/csrc/selective_scan/static_switch.h": "1eb3ebf9f2f231be1bfad83ad15612031e24980336c0019f2d742bce599f704c",
  "selective_scan/csrc/selective_scan/reverse_scan.cuh": "4b770c5890104d771f9c22ae56767700dc76d5d71b81cee70e2b88d54d2c315b",
  "selective_scan/csrc/selective_scan/cub_extra.cuh": "2735eb386be604b6f97fc45ffd59c89ca205129eedb25e5b1399d03be797b05e",
  "selective_scan/csrc/selective_scan/uninitialized_copy.cuh": "80d81db2fe2c47d0e89b9b29643fda533ed065d2157d9a1c6a41a1078c057de6",
  "selective_scan/csrc/selective_scan/cus/selective_scan.cpp": "36617cb69e093441742a07a4090b8bb1fc4b84b1d5f810176e5ff5b9c5f49b05",
  "selective_scan/csrc/selective_scan/cus/selective_scan_core_fwd.cu": "ac11838b57e65cb920d3f4dc7f2a816b6c46c68708a1d6ba107050378e8583ce",
  "selective_scan/csrc/selective_scan/cus/selective_scan_core_bwd.cu": "02a11cfa68a7ba229ad6cab4115d0c1607fa2890c0393212f3edf59695e0edb3",
  "selective_scan/csrc/selective_scan/cus/selective_scan_fwd_kernel.cuh": "462836f2391ed56ee1c2f314915bc0fced20541fd212a86ff668b10fa9984127",
  "selective_scan/csrc/selective_scan/cus/selective_scan_bwd_kernel.cuh": "acb418bfe8f22a2a03cc12751b40133f67953c4a86db586e3dc8104bfb16d120"
}
```

## Adaptations

Windows compiler flags and independent packaging; explicit host/device headers;
portable pointer qualifiers and log2(e) constant; explicit dependent-type aliases;
qualified reverse-scan type name; explicit boolean template dispatch in launch
helpers to avoid NVCC/MSVC nested-lambda constant-expression errors.
Scan mathematics, chunking, launch choices,
fast math, and accumulation types are preserved.

## NVIDIA notice

```text
/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
```
