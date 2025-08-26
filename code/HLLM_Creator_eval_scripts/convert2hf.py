# Copyright 2025 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src_path", type=str, default="")
parser.add_argument("--trg_path", type=str, default="")
args = parser.parse_args()


def run(args):
    state_dict = {}
    pretrain_state_dict = torch.load(args.src_path, map_location="cpu")
    for k, v in pretrain_state_dict.items():
        if k.startswith('creative_llm.'):
            print(k, v.size())
            k = '.'.join(k.split('.')[1:])
            state_dict[k] = v
    torch.save(state_dict, args.trg_path)


if __name__ == "__main__":
    run(args)
