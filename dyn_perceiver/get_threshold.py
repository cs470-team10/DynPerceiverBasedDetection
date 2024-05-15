from mmengine.runner.amp import autocast
from torch import Tensor
import torch
import torch.nn as nn
import math
from cs470_logger.cs470_print import cs470_print

def generate_distribution(each_exit = False) -> Tensor:
    probs_list = []
    if each_exit:
        for i in range(4):
            probs = torch.zeros(4, dtype=torch.float)
            probs[i] = 1
            probs_list.append(probs)
    else:
        p_list = torch.zeros(34)
        for i in range(17):
            p_list[i] = (i + 4) / 20
            p_list[33 - i] = 20 / (i + 4)
            
        k = [0.85, 1, 0.5, 1]
        for i in range(33):
            probs = torch.exp(torch.log(p_list[i]) * torch.range(1, 4))
            probs /= probs.sum()
            for j in range(3):
                probs[j] *= k[j]
                probs[j+1:4] = (1 - probs[0:j+1].sum()) * probs[j+1:4] / probs[j+1:4].sum()
            probs_list.append(probs)
    return probs_list # size : 34 * 4

def get_threshold(model, val_loader, fp16: bool):
    with autocast(enabled=fp16):
        #val_loader.batch_size = 128
        tester = Tester(model)
        
        val_pred, val_target = tester.calc_logit(val_loader, early_break = True)
        
        probs_list = generate_distribution()
        
        return_list = []
        for probs in probs_list:
            print("\n")
            cs470_print('*****************')
            cs470_print(str(probs))
            acc_val, T = tester.dynamic_eval_find_threshold(val_pred, val_target, probs)
            return_list.append(T)
            cs470_print(str(T))
            cs470_print('valid acc: {:.3f}'.format(acc_val))
        
    cs470_print('----------ALL DONE-----------')
    return_list.append(torch.tensor([1,1,1,-1]))
    #print("Threshold list(get_threshold.py) :", return_list)
    return return_list

class Tester(object):
    def __init__(self, model):
        # self.args = args
        self.model = model
        self.softmax = nn.Softmax(dim=1).cuda()

    def calc_logit(self, dataloader, early_break=False, max_images = 5000):
        
        self.model.backbone.eval()
        self.model.cuda()
        n_stage = 4
        logits = [[] for _ in range(n_stage)]
        targets = []
        
        for idx, sample in enumerate(dataloader):
            if early_break and idx > max_images:
                break
            
            target = sample['data_samples']
            
            tmp_labels = []
            
            for t in target:
                labels = t.gt_instances.labels.tolist()
                tmp_labels.append(labels[0])
            
            targets.append(torch.tensor(tmp_labels))
            
            #packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])
            #data = self.data_preprocessor(data_batch)
            data = self.model.data_preprocessor(sample)
            input = data['inputs']

            input = input.cuda()
            with torch.no_grad():
                _x, y_early3, y_att, y_cnn, y_merge = self.model.backbone(input)
                output = [y_early3, y_att, y_cnn, y_merge]
                for b in range(n_stage):
                    _t = self.softmax(output[b])

                    logits[b].append(_t)
                    
            if idx % 50 == 0:
                cs470_print('Generate Logit: [{0}/{1}]'.format(idx, max(max_images, len(dataloader))))
        
        for b in range(n_stage):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_stage, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_stage):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)

        return ts_logits, ts_targets

    def dynamic_eval_find_threshold(self, logits, targets, p):
        n_stage, n_sample, c = logits.size()
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False)

        _, sorted_idx = max_preds.sort(dim=1, descending=True)

        filtered = torch.zeros(n_sample)

        T = torch.Tensor(n_stage).fill_(1e8)

        for k in range(n_stage - 1):
            acc, count = 0.0, 0
            out_n = math.floor(n_sample * p[k])
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = max_preds[k][ori_idx]
                        break
            filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))

        T[n_stage -1] = -1e8 # accept all of the samples at the last stage

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage) # 각 stage에서 correctly classified된 sample수와 총 검사한 sample 수
        acc = 0 # 전체 system의 accuracy와 FLOPs
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force the sample to exit at k
                    if int(gold_label.item()) == int(argmax_preds[k][i].item()):
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0
        for k in range(n_stage):
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, T