import wandb
import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
import anndata as ad
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from tqdm import tqdm
from copy import deepcopy
from ..utils.eval import downstream_eval, aggregate_eval_results
from ..utils.data import XDict, TranscriptomicDataset
from typing import List, Literal, Union
from .experimental import symbol_to_ensembl
from torch.utils.data import DataLoader
import warnings
from . import Pipeline, load_pretrain
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

CellEmbeddingDefaultModelConfig = {
    'dec_mod': 'nbmlp',  # 디코더 방식
    'latent_mod': 'vae',  # 잠재 공간 모듈
    'objective': 'nb',
    'mask_node_rate': 0.25,
    'mask_feature_rate': 0.1,
    'max_batch_size': 70000,
}

CellEmbeddingDefaultPipelineConfig = {
    'es': 20,
    'lr': 5e-4,
    'wd': 1e-6,
    'scheduler': 'plat',
    'epochs': 10,
    'max_eval_batch_size': 128, #100000
    'patience': 5,
    'workers': 0,
}

CellEmbeddingWandbConfig = {
    "mode":"offline",  # 인터넷 없이 로깅
    "entity": "juha95-university-of-manchester",  # 엔티티(팀) 이름
    "project": "CellEmbedding",  # 프로젝트 이름
    "config": {  # 하이퍼파라미터 정보
        **CellEmbeddingDefaultModelConfig,
        **CellEmbeddingDefaultPipelineConfig
    },
}

def inference(model, dataloader, split, device, batch_size, order_required=False):
    if order_required and split:
        warnings.warn('When cell order required to be preserved, dataset split will be ignored.')

    with torch.no_grad():
        model.eval()
        epoch_loss = []
        order_list = []
        pred = []
        for i, data_dict in enumerate(dataloader):
            if not order_required and split and np.sum(data_dict['split'] == split) == 0:
                continue

            idx = torch.arange(data_dict['x_seq'].shape[0])
            for j in range(0, len(idx), batch_size):
                if len(idx) - j < batch_size:
                    cur = idx[j:]
                else:
                    cur = idx[j:j + batch_size]
                input_dict = {}
                for k in data_dict:
                    if k == 'x_seq':
                        input_dict[k] = data_dict[k].index_select(0, cur).to(device)
                    elif k == 'gene_mask':
                        input_dict[k] = data_dict[k].to(device)
                    elif k not in ['gene_list', 'split']:
                        input_dict[k] = data_dict[k][cur].to(device)

                if 'gene_mask' not in input_dict:
                    input_dict['gene_mask'] = torch.arange(input_dict['x_seq'].shape[1]).to(device)

                x_dict = XDict(input_dict)
                out_dict, loss = model(x_dict, data_dict['gene_list']) #
                epoch_loss.append(loss.item()) #
                pred.append(out_dict['pred'])
                if order_required:
                    order_list.append(input_dict['order_list'])
        pred = torch.cat(pred)
        if order_required:
            order = torch.cat(order_list)
            order.scatter_(0, order.clone(), torch.arange(order.shape[0]).to(order.device))
            pred = pred[order]

        return {'pred': pred,
                'loss': sum(epoch_loss) / len(epoch_loss)}


class CellEmbeddingPipeline(Pipeline):
    def __init__(self,
                 pretrain_prefix: str,
                 overwrite_config: dict = CellEmbeddingDefaultModelConfig,
                 pretrain_directory: str = './ckpt',
                 ):
        super().__init__(pretrain_prefix, overwrite_config, pretrain_directory)
        self.label_encoders = None
        # self.pretrain_prefix = pretrain_prefix
        # self.pretrain_directory = pretrain_directory

    def fit(self, adata: ad.AnnData,
            train_config: dict = None,
            split_field: str = None,
            wandb_config: dict = None,
            train_split: str = 'train',
            valid_split: str = 'valid',
            covariate_fields: List[str] = None,
            label_fields: List[str] = None,
            batch_gene_list: dict = None,
            ensembl_auto_conversion: bool = True,
            # A bool value indicating whether the function automativally convert symbols to ensembl id
            device: Union[str, torch.device] = 'cpu'
            ):
        config = CellEmbeddingDefaultPipelineConfig.copy()
        if train_config:
            config.update(train_config)
        self.model.to(device)
        assert not self.fitted, 'Current pipeline is already fitted and does not support continual training. Please initialize a new pipeline.'
        if label_fields:
            warnings.warn('`label_fields` argument is ignored in CellEmbeddingPipeline.')
        if covariate_fields:
            warnings.warn('`covariate_fields` argument is ignored in CellEmbeddingPipeline.')
        if batch_gene_list:
            warnings.warn('`batch_gene_list` argument is ignored in CellEmbeddingPipeline.')
    
        # _model = load_pretrain(self.pretrain_prefix, {}, self.pretrain_directory)
        # _model.to(device)
    
        adata = self.common_preprocess(adata, 0, None, ensembl_auto_conversion)
        print(f'After filtering, {adata.shape[1]} genes remain.')
        dataset = TranscriptomicDataset(adata, split_field)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=config['workers'])
        optim = torch.optim.AdamW(self.model.parameters(), lr=config['lr'], weight_decay=config['wd'])
        
        if config['scheduler'] == 'plat':
            scheduler = ReduceLROnPlateau(optim, 'min', patience=config['patience'], factor=0.9)
        else:
            scheduler = None

        if wandb_config is not None:  # W&B 설정이 제공된 경우
            run = wandb.init(**wandb_config)  # 실험 시작
    
        train_loss = []
        valid_loss = []
        best_dict = None
    
        for epoch in tqdm(range(config['epochs'])):
            self.model.train()
            epoch_loss = []
    
            if epoch < 5:
                for param_group in optim.param_groups:
                    param_group['lr'] = config['lr'] * (epoch + 1) / 5
    
            for i, data_dict in enumerate(dataloader):
                if split_field and np.sum(data_dict['split'] == train_split) == 0:
                    continue
                input_dict = data_dict.copy()
                del input_dict['gene_list'], input_dict['split']
                for k in input_dict:
                    input_dict[k] = input_dict[k].to(device)
                
                if 'gene_mask' not in input_dict:
                    input_dict['gene_mask'] = torch.arange(input_dict['x_seq'].shape[1]).to(device) ##

                x_dict = XDict(input_dict)
                out_dict, loss = self.model(x_dict, data_dict['gene_list'])
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                optim.step()
                epoch_loss.append(loss.item())
    
            train_loss.append(sum(epoch_loss) / len(epoch_loss))
            if config['scheduler'] == 'plat':
                scheduler.step(train_loss[-1])
            valid_l = inference(self.model, dataloader, valid_split, device, config['max_eval_batch_size'])
            valid_loss.append(valid_l['loss'])

            print(f'Epoch {epoch} | Train loss: {train_loss[-1]:.4f} | Valid loss: {valid_loss[-1]:.4f}')
            
            if min(valid_loss) == valid_loss[-1]:
                best_dict = deepcopy(self.model.state_dict())
            
            if wandb_config is not None:  # W&B 사용 시 로그 기록
                run.log({
                    # "epoch": epoch,
                    "train_loss": train_loss[-1],
                    "valid_loss": valid_loss[-1],
                })
        if wandb_config is not None:
            run.finish()  # 실험 종료 후 마무리
    
        assert best_dict, 'Best state dict was not stored. Please report this issue on Github.'
        self.model.load_state_dict(best_dict)
    
        # embed_state = self.model.state_dict()
        # for k, v in self.model.state_dict().items():
        #     if k in embed_state and embed_state[k].shape == v.shape:
        #         embed_state[k] = v
        # self.model.load_state_dict(embed_state)
        self.fitted = True
        return self

    def predict(self, adata: ad.AnnData,
                inference_config: dict = None,
                covariate_fields: List[str] = None,
                batch_gene_list: dict = None,
                ensembl_auto_conversion: bool = True,
                device: Union[str, torch.device] = 'cpu'
                ):
        self.model.to(device)
        config = CellEmbeddingDefaultPipelineConfig.copy()
        if inference_config:
            config.update(inference_config)
        adata = self.common_preprocess(adata, 0, covariate_fields=None, ensembl_auto_conversion=ensembl_auto_conversion)
        print(f'After filtering, {adata.shape[1]} genes remain.')
        dataset = TranscriptomicDataset(adata, None, order_required=True)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

        if inference_config and 'batch_size' in inference_config:
            batch_size = inference_config['batch_size']
        else:
            batch_size = 0

        if batch_size <= 0:
            batch_size = adata.shape[0]

        if covariate_fields:
            warnings.warn('`covariate_fields` argument is ignored in CellEmbeddingPipeline.')
        if batch_gene_list:
            warnings.warn('`batch_gene_list` argument is ignored in CellEmbeddingPipeline.')
        return inference(self.model, dataloader, None, device,
                  config['max_eval_batch_size'], order_required=True)['pred']
    # def _inference(self, adata: ad.AnnData,
    #             batch_size: int = 0,
    #             device: Union[str, torch.device] = 'cpu',
    #             ensembl_auto_conversion: bool = True):
    #     self.model.to(device)
    #     adata = self.common_preprocess(adata, 0, covariate_fields=None, ensembl_auto_conversion=ensembl_auto_conversion)
    #     print(f'After filtering, {adata.shape[1]} genes remain.')
    #     dataset = TranscriptomicDataset(adata, order_required=True)
    #     dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
        
    #     order_list = []
    #     if batch_size <= 0:
    #         batch_size = adata.shape[0]

    #     with torch.no_grad():
    #         self.model.eval()
    #         pred = []
    #         for i, data_dict in enumerate(dataloader):
    #             idx = torch.arange(data_dict['x_seq'].shape[0])
    #             for j in range(0, len(idx), batch_size):
    #                 if len(idx) - j < batch_size:
    #                     cur = idx[j:]
    #                 else:
    #                     cur = idx[j:j + batch_size]
    #                 input_dict = {}
    #                 for k in data_dict:
    #                     if k == 'x_seq':
    #                         input_dict[k] = data_dict[k].index_select(0, cur).to(device)
    #                     elif k not in ['gene_list', 'split']:
    #                         input_dict[k] = data_dict[k][cur].to(device)
    #                 x_dict = XDict(input_dict)
    #                 out_dict, _ = self.model(x_dict, data_dict['gene_list'])
    #                 order_list.append(input_dict['order_list'])
    #                 pred.append(out_dict['pred'])#[input_dict['order_list']])
    #         order = torch.cat(order_list)
    #         order.scatter_(0, order.clone(), torch.arange(order.shape[0]).to(order.device))
    #         pred = torch.cat(pred)
    #         pred = pred[order]
    #         return pred

    def score(self, adata: ad.AnnData,
              evaluation_config: dict = None,
              split_field: str = None,
              target_split: str = 'test',
              covariate_fields: List[str] = None,
              label_fields: List[str] = None,
              batch_gene_list: dict = None,
              ensembl_auto_conversion: bool = True,
              device: Union[str, torch.device] = 'cpu'
              ):
        
        self.model.to(device)
        config = CellEmbeddingDefaultPipelineConfig.copy()
        if evaluation_config:
            config.update(evaluation_config)
        adata = self.common_preprocess(adata, 0, covariate_fields=None, ensembl_auto_conversion=ensembl_auto_conversion)
        print(f'After filtering, {adata.shape[1]} genes remain.')
        dataset = TranscriptomicDataset(adata, None, order_required=True)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

        
        if evaluation_config and 'batch_size' in evaluation_config:
            batch_size = evaluation_config['batch_size']
        else:
            batch_size = 0
        if len(label_fields) != 1:
            raise NotImplementedError(
                f'`label_fields` containing multiple labels (f{len(label_fields)}) is not implemented for evaluation of cell embedding pipeline. Please raise an issue on Github for further support.')
        if split_field:
            warnings.warn('`split_field` argument is ignored in CellEmbeddingPipeline.')
        if target_split:
            warnings.warn('`target_split` argument is ignored in CellEmbeddingPipeline.')
        if covariate_fields:
            warnings.warn('`covariate_fields` argument is ignored in CellEmbeddingPipeline.')
        if batch_gene_list:
            warnings.warn('`batch_gene_list` argument is ignored in CellEmbeddingPipeline.')

        adata = adata.copy()
        pred = inference(self.model, dataloader, None, device,
                  config['max_eval_batch_size'], order_required=True)['pred']
        adata.obsm['emb'] = pred.cpu().numpy()

        if len(label_fields) != 1:
            raise NotImplementedError(
                f'`label_fields` containing multiple labels (f{len(label_fields)}) is not implemented for imputation pipeline. Please raise an issue on Github for further support.')
        
        if 'method' in evaluation_config and evaluation_config['method'] == 'rapids':
            sc.pp.neighbors(adata, use_rep='emb', method='rapids')
        else:
            sc.pp.neighbors(adata, use_rep='emb')
        best_ari = -1
        best_nmi = -1
        for res in range(1, 15, 1):
            res = res / 10
            if 'method' in evaluation_config and evaluation_config['method'] == 'rapids':
                import rapids_singlecell as rsc
                rsc.tl.leiden(adata, resolution=res, key_added='leiden')
            else:
                sc.tl.leiden(adata, resolution=res, key_added='leiden')
            ari_score = adjusted_rand_score(adata.obs['leiden'].to_numpy(), adata.obs[label_fields[0]].to_numpy())
            if ari_score > best_ari:
                best_ari = ari_score
            nmi_score = normalized_mutual_info_score(adata.obs['leiden'].to_numpy(), adata.obs[label_fields[0]].to_numpy())
            if nmi_score > best_nmi:
                best_nmi = nmi_score
        return {'ari': best_ari, 'nmi': best_nmi}




