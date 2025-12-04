

import torch
from torch.nn import functional as F

from models.utils.cl2branches import CLModel2Branches
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, add_saliency_args, ArgumentParser
from utils.buffer import Buffer

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_saliency_args(parser)
    
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    
    return parser

class DerppSER(CLModel2Branches):
    
    NAME = 'derpp_ser'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform) -> None:
        super(DerppSER, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)


    def observe(self, inputs, labels, not_aug_inputs, current_task_labels, task_number=-1, args=None, tb_logger=None, epoch=-1):

        self.opt.zero_grad()

        if self.saliency_net is not None and not self.args.saliency_frozen:
            self.saliency_opt.zero_grad()
        if self.args.saliency_frozen:
            assert not self.saliency_net.training
        
        assert isinstance(inputs, list)
        imgs, sal_maps = inputs
        B = labels.size(0)

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
            imgs_real     = imgs[real_idx]
            sal_maps_real = sal_maps[real_idx]

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

            # forward "pulito": ResNet senza adapter, senza saliency_net
            logits_dream = self.forward_no_saliency(imgs_dream, returnt='out')
            logits_all[dream_idx] = logits_dream

        cls_loss = self.loss(logits_all, labels)

        # --------------------------------------------------
        # 5) Replay DERPP (MSE su logits + CE da buffer)
        # --------------------------------------------------
        loss_distill = torch.tensor(0.0, device=labels.device)  # α * MSE
        loss_buf_ce  = torch.tensor(0.0, device=labels.device)  # β * CE buffer
        if hasattr(self, 'buffer') and not self.buffer.is_empty():
            saliency_status = self.saliency_net.training
            self.saliency_net.eval()
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=None)

            with torch.no_grad():
                _, sal_features = self.saliency_net(buf_inputs)

            buf_outputs = self.forward_mnp(buf_inputs, sal_features)
            
            loss_distill = F.mse_loss(buf_outputs, buf_logits)
            loss_distill = self.args.alpha * loss_distill

            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=None)
            with torch.no_grad():
                _, sal_features = self.saliency_net(buf_inputs)
            
            buf_outputs = self.forward_mnp(buf_inputs, sal_features)

            loss_buf_ce = self.loss(buf_outputs, buf_labels)
            loss_buf_ce = self.args.beta * loss_buf_ce
            
            self.saliency_net.train(saliency_status)

        
        total_loss += cls_loss + loss_distill + loss_buf_ce
        self.opt.step()

        # filters the items in the batch based on the current task
        if current_task_labels != []:
            mask_list = torch.stack([labels == l for l in current_task_labels])
            mask = torch.any(mask_list, dim = 0)
            not_aug_inputs = not_aug_inputs[mask]
            labels = labels[mask]
            logits_all = logits_all[mask]
    

        if hasattr(self, 'buffer'):
            self.buffer.add_data(examples=not_aug_inputs,
                             labels = labels,
                             logits = logits_all.data)
        return [total_loss.item(), sal_loss.item() if real_idx.numel() > 0 else 0.0]
