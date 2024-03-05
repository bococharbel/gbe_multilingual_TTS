# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import argparse

def parse_hifigan_args(parent, add_help=False):
    """
    Parse commandline arguments.
    """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help, allow_abbrev=False)

    # misc parameters
    #parser.add_argument('--n-mel-channels', default=80, type=int,
    #                    help='Number of bins in mel-spectrograms')

    parser.add_argument('--num-mels', default=80, type=int,
                        help='Number of bins in mel-spectrograms')

    # hifigan params
    parser.add_argument('--resblock', default=1, type=int,   help='Number of resblock')
    parser.add_argument('--num-gpus', default=0, type=int,   help='Number of GPU ')

    parser.add_argument('--adam-b1', default=0.8, type=int,   help='Adam ')
    parser.add_argument('--adam-b2', default=0.99, type=int,   help='Number of GPU ')
    parser.add_argument('--lr-decay', default=0.999, type=int,   help='Number of GPU ')

    parser.add_argument("--upsample-rates", nargs="+", default=[8,8,2,2])
    parser.add_argument("--upsample-kernel-sizes", nargs="+", default=[16,16,4,4])
    parser.add_argument("--upsample-initial-channel", type=int,  default=512)
    parser.add_argument("--resblock-kernel-sizes", nargs="+", default=[3,7,11])
    parser.add_argument("--resblock-dilation-sizes", nargs="+", default=[[1,3,5], [1,3,5], [1,3,5]])

    parser.add_argument('--segment-size', default=8192, type=int,
                        help='Segment length (audio samples) processed per iteration')

    parser.add_argument('--num-freq', default=1025, type=int,   help='')
    parser.add_argument('--n-fft', default=1024, type=int,   help='')

    

    return parser
