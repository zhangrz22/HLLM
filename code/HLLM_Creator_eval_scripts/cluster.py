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
import argparse
import glob

import torch

# pip3 install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# pip3 install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
import faiss

parser = argparse.ArgumentParser()
parser.add_argument("--user_emb_path", type=str)
parser.add_argument("--num_clusters", type=int, default=256)
args = parser.parse_args()

embs = []
for file in glob.glob(f'{args.user_emb_path}/rank*.pt'):
    emb = torch.load(file, map_location='cpu').float().cpu()
    print(f"{emb.size() = }")
    embs.append(emb)

embs = torch.cat(embs, dim=0)
print(embs.size())
embeddings_np = embs.float().cpu().numpy()
kmeans = faiss.Kmeans(
    embs.size(1), args.num_clusters, gpu=True, verbose=True, niter=40, nredo=2
)
kmeans.train(embs)
centors = torch.from_numpy(kmeans.centroids)
print(centors.size())
torch.save(centors, f'{args.user_emb_path}/cluster_{args.num_clusters}.pt')
