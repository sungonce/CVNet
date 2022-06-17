# written by Seongwon Lee (won4113@yonsei.ac.kr)

import os
import torch
import numpy as np

from tqdm import tqdm

from test.config_gnd import config_gnd
from test.test_utils import extract_feature, rerank_ranks_revisitop, test_revisitop
from test.dataset import DataSet

@torch.no_grad()
def test_model(model, data_dir, dataset_list, scale_list, topk_list):
    torch.backends.cudnn.benchmark = False
    model.eval()
    for dataset in dataset_list:
        text = '>> {}: Global Retrieval for scale {} with CVNet-Global'.format(dataset, str(scale_list))
        print(text)
        if dataset == 'roxford5k':
            gnd_fn = 'gnd_roxford5k.pkl'
        elif dataset == 'rparis6k':
            gnd_fn = 'gnd_rparis6k.pkl'
        else:
            assert dataset
        print("extract query features")
        Q = extract_feature(model, data_dir, dataset, gnd_fn, "query", scale_list)
        print("extract database features")
        X = extract_feature(model, data_dir, dataset, gnd_fn, "db", scale_list)

        cfg = config_gnd(dataset,data_dir)

        # perform search
        print("perform global retrieval")
        sim = np.dot(X, Q.T)
        ranks = np.argsort(-sim, axis=0)

        # revisited evaluation
        gnd = cfg['gnd']
        ks = [1, 5, 10]
        (mapE, apsE, mprE, prsE), (mapM, apsM, mprM, prsM), (mapH, apsH, mprH, prsH) = test_revisitop(cfg, ks, [ranks, ranks, ranks])

        print('Global retrieval results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))

        print('>> {}: Reranking results with CVNet-Rerank'.format(dataset))

        query_dataset = DataSet(data_dir, dataset, gnd_fn, "query", [1.0])
        db_dataset = DataSet(data_dir, dataset, gnd_fn, "db", [1.0])
        sim_corr_dict = {}
        for topk in topk_list:
            print("current top-k value: ", topk)
            for i in tqdm(range(int(cfg['nq']))):
                im_q = query_dataset.__getitem__(i)[0]
                im_q = torch.from_numpy(im_q).cuda().unsqueeze(0)
                feat_q = model.extract_featuremap(im_q)

                rerank_count = np.zeros(3, dtype=np.uint16)
                for j in range(int(cfg['n'])):
                    if (rerank_count >= topk).sum() == 3:
                        break

                    rank_j = ranks[j][i]

                    if rank_j in gnd[i]['junk']:
                        continue
                    elif rank_j in gnd[i]['easy']:
                        append_j = np.asarray([True, True, False])
                    elif rank_j in gnd[i]['hard']:
                        append_j = np.asarray([False, True, True])
                    else: #negative
                        append_j = np.asarray([True, True, True])

                    append_j *= (rerank_count < topk)

                    if append_j.sum() > 0:
                        im_k = db_dataset.__getitem__(rank_j)[0]
                        im_k = torch.from_numpy(im_k).cuda().unsqueeze(0)
                        feat_k = model.extract_featuremap(im_k)

                        score = model.extract_score_with_featuremap(feat_q, feat_k).cpu()
                        sim_corr_dict[(rank_j, i)] = score
                        rerank_count += append_j
    
            mix_ratio = 0.5
            ranks_corr_list = rerank_ranks_revisitop(cfg, topk, ranks, sim, sim_corr_dict, mix_ratio)
            (mapE_r, apsE_r, mprE_r, prsE_r), (mapM_r, apsM_r, mprM_r, prsM_r), (mapH_r, apsH_r, mprH_r, prsH_r) = test_revisitop(cfg, ks, ranks_corr_list)
            print('Reranking results: mAP E: {}, M: {}, H: {}'.format(np.around(mapE_r*100, decimals=2), np.around(mapM_r*100, decimals=2), np.around(mapH_r*100, decimals=2)))
        
    torch.backends.cudnn.benchmark = True

