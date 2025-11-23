# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from datasets import get_dataset

from models.utils.cl2branches import CLModel2Branches
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, add_saliency_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_saliency_args(parser)

    return parser


class ErACESER(CLModel2Branches):
    NAME = 'er_ace_ser'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ErACESER, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0

    def end_task(self, dataset):
        self.task += 1


    def observe(self, inputs, labels, not_aug_inputs, current_task_labels):
    
        self.opt.zero_grad()
        if self.saliency_net is not None and not self.args.saliency_frozen:
            self.saliency_opt.zero_grad()

        if self.args.saliency_frozen:
            assert not self.saliency_net.training

        # er-ace logic
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()
    
    
        assert isinstance(inputs, list)
        imgs, sal_maps = inputs  # imgs: (B, C, H, W) (8, 3, 288, 234), sal_maps: (B, 1, H, W)(8, 1, 288, 234)
        B = labels.size(0) # labels: tensor di shape (B,)

        # current_task_labels: lista o tensor di etichette REAL del task corrente
        current_task_labels_tensor = torch.as_tensor(current_task_labels, device=labels.device)

        # is_real[b] = True se labels[b] è una label del task REAL corrente
        is_real = (labels.unsqueeze(1) == current_task_labels_tensor.unsqueeze(0)).any(dim=1)

        real_idx = is_real.nonzero(as_tuple=True)[0]        # indici REAL
        dream_idx = (~is_real).nonzero(as_tuple=True)[0]    # indici DREAM

        logits_all = torch.zeros((B, self.num_classes), device=labels.device)  
        sal_loss = torch.tensor(0., device=labels.device)  
        total_loss = torch.tensor(0., device=labels.device)

        # --------------------------------------------------
        # 1) REAL: saliency_net + MNP (SER)
        # --------------------------------------------------
        # Sottobatch REAL (hanno sal_map ground-truth reale)
        if real_idx.numel() > 0:
            imgs_real      = imgs[real_idx]
            sal_maps_real  = sal_maps[real_idx]
            # labels_real    = labels[real_idx]

            if not self.args.saliency_frozen:
                sal_pred, sal_features = self.saliency_net(imgs_real)
            else:
                with torch.no_grad():
                    sal_pred, sal_features = self.saliency_net(imgs_real)

            sal_features = [sal_f.detach() for sal_f in sal_features]
            logits_real = self.forward_mnp(imgs_real, sal_features)
            
            logits_all[real_idx] = logits_real

            sal_loss = self.sal_criterion(sal_pred, sal_maps_real) * self.sal_coeff 
            if not self.args.saliency_frozen:
                sal_loss.backward()
                self.saliency_opt.step()

        # --------------------------------------------------
        # 2) DREAM: backbone senza salienza nè MNP
        # --------------------------------------------------
        # Sottobatch DREAM (sal_map è dummy, da ignorare) -> forward classico
        if dream_idx.numel() > 0:
            imgs_dream   = imgs[dream_idx]
            # labels_dream = labels[dream_idx]

            # forward "pulito": ResNet senza adapter, senza saliency_net
            logits_dream = self.forward_no_saliency(imgs_dream, returnt='out')
            logits_all[dream_idx] = logits_dream

        # --------------------------------------------------
        # 3) ER-ACE: masking dei logits del batch corrente
        # --------------------------------------------------
        mask = torch.zeros_like(logits_all)
        mask[:, present] = 1

        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.task > 0:
            logits_all = logits_all.masked_fill(mask == 0, torch.finfo(logits_all.dtype).min)

        cls_loss = self.loss(logits_all, labels)

        # --------------------------------------------------
        # 4) Replay da buffer (ER)
        # --------------------------------------------------
        loss_re = torch.tensor(0., device=labels.device)

        if self.task > 0:
            # sample from buffer: it contains only past real task data with saliency maps
            saliency_status = self.saliency_net.training
            self.saliency_net.eval()
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=None)

            with torch.no_grad():
                _, sal_features = self.saliency_net(buf_inputs)

            buf_outputs = self.forward_mnp(buf_inputs, sal_features)

            loss_re = self.loss(buf_outputs, buf_labels)
            self.saliency_net.train(saliency_status)

        # --------------------------------------------------
        # 5) Backward
        # --------------------------------------------------
        total_loss += cls_loss + loss_re
        total_loss.backward()
        self.opt.step()

        # --------------------------------------------------
        # 6) Aggiornamento buffer (WSCL-style)
        # --------------------------------------------------
        # filters the items in the batch based on the current task
        if current_task_labels != []:
            # build a mask that is True for samples whose label is in current_task_labels
            mask_list = torch.stack([labels == l for l in current_task_labels])
            mask = torch.any(mask_list, dim=0)

            not_aug_inputs = not_aug_inputs[mask]
            labels = labels[mask]
        
        # we add to the buffer only the REAL samples of the current batch
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return [total_loss.item(), sal_loss.item() if real_idx.numel() > 0 else 0.0]
