################################################################################
# Starlab Transformer Compression with PET (Parameter-Efficient Knowledge Distillation on Transformer)
#
# Author: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# Version : 1.0
# Date : Nov 29, 2022
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
# This code is mainly based on the [GitHub Repository]
# [GitHub Repository]: https://github.com/facebookresearch/fairseq
################################################################################
import math
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import II

@dataclass
class PETLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
    default=0.0,
    metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
    default=False,
    metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
    default=0,
    metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")

def distillation_loss_kd_no_torch(st_output, t_output, T=10, reduction_kd='batchmean'):
    """
     customized function to compute the distillation loss
     1. Goal: Learn logits from the last layer
     2. st_output, t_output -(tuple)
        st_output[0] - (Tensor): output from each decoder ([batch, tgt_len, embed_dim])
     3. T: how much to rely on the teacher's soft predictions
        (higher T refers to more diverse probability dist. over classes)
     4. for all data, sum ( multiplication of teacher's and student's output probability for each class no matter
                            whether it is the true label or not ) => apply to both stu and teacher
    """
    if t_output is not None:
        d_loss=nn.KLDivLoss(reduction=reduction_kd)(F.log_softmax(st_output[0]/T, dim=2),
                                F.softmax(t_output[0]/T, dim=1)
                             )*T*T
        target = F.softmax(t_output[0]/T, dim=2)
        s_input = F.log_softmax(st_output[0]/T, dim=2)
        elem_mul_logit = target * s_input
        d_loss = (-1) * torch.sum(elem_mul_logit)
    else:
        d_loss = 0.0
    return d_loss

def patience_loss_kd_no_torch(st_output,st_output_en, t_output,t_output_en, normalized_patience=False, distill_type="skip"):

    """
    customized fuction to compute the patience loss
    distill_type ="skip" / "last"
    Given teacher: {1,2,3,4,5,6}, student: {1,2,3,4}
    if distill_type == "skip":
        1 -> 1
        3 -> 2
        5 -> 3
    else:
        3 -> 1
        4 -> 2
        5 -> 3
    """
    #1. inter_states without the last states (List[Tensor])
    st_dec_inter_states = torch.stack(st_output[1]["inner_states"][:-2])
    st_enc_inter_states = torch.stack(st_output_en["encoder_states"][:-2])

    t_dec_inter_states = torch.stack(t_output[1]["inner_states"][:-2])
    t_enc_inter_states = torch.stack(t_output_en["encoder_states"][:-2])

    #2. extract states of the [CLS] tokens
    st_enc_cls_inter_states = st_enc_inter_states[:,-1,:,:] #[n_layers, src_len, batch, enc_dim]
    st_dec_cls_inter_states = st_dec_inter_states[:,-1,:,:] #[n_layers, tgt_len, batch, enc_dim]
    mapped_t_layers = [0, 2, 4] if distill_type == "skip" else [2, 3, 4]
    t_enc_cls_inter_states = t_enc_inter_states[mapped_t_layers, -1, :, :] #[n_layers, src_len, batch, enc_dim]
    t_dec_cls_inter_states = t_dec_inter_states[mapped_t_layers, -1, :, :] #[n_layers, tgt_len, batch, enc_dim]

    #3. normalize each inter. state
    if normalized_patience:
        #3.1. encoder
        t_enc_cls_inter_states = F.normalize(t_enc_cls_inter_states, p=2, dim=2)
        st_enc_cls_inter_states = F.normalize(st_enc_cls_inter_states, p=2, dim=2)
        #3.2. decoder
        t_dec_cls_inter_states = F.normalize(t_dec_cls_inter_states, p=2, dim=2)
        st_dec_cls_inter_states = F.normalize(st_dec_cls_inter_states, p=2, dim=2)

    encoder_pt_loss = F.mse_loss(t_enc_cls_inter_states.float(), st_enc_cls_inter_states.float())
    decoder_pt_loss = F.mse_loss(t_dec_cls_inter_states.float(), st_dec_cls_inter_states.float())
    return (encoder_pt_loss + decoder_pt_loss) * 0.5

def distillation_loss_kd(st_output, t_output, T=10,reduction_kd='batchmean'):
    "compute distillation loss"
    if t_output is not None:
        d_loss=nn.KLDivLoss(reduction=reduction_kd)(  F.log_softmax(st_output[0]/T, dim=1),
                                F.softmax(t_output[0]/T, dim=1)
                             )*T*T
    else:
        d_loss = 0.0
    return d_loss

def patience_loss_kd(st_output,st_output_en, t_output,t_output_en, normalized_patience=False):
    "compute patience loss"

    st_attn=st_output[1]["inner_states"] #attn list from extra states

    st_patience_en=st_output_en["encoder_states"] #list of the encoder states [T,B,C]
    t_attn=t_output[1]["inner_states"]

    t_patience_en=t_output_en["encoder_states"]  

    st_patience=torch.stack(st_patience_en[:-1])[:,:,-1,:] #1
    t_patience=torch.stack(t_patience_en[len(t_patience_en[:-1])-len(st_patience_en[:-1]):-1])[:,:,-1,:] #6 pkd_last

    t_attn=torch.stack(t_attn[len(t_attn)-len(st_attn):-1])[:,:,-1,:] # pkd-last
    s_attn=torch.stack(st_attn[:-1])[:,:,-1,:]

    if normalized_patience:
        t_patience=F.normalize(t_patience,p=2,dim=2)
        st_patience=F.normalize(st_patience,p=2,dim=2)

        ## distill decoder
        t_attn=F.normalize(t_attn,p=2,dim=2)
        s_attn=F.normalize(s_attn,p=2,dim=2)

    return (F.mse_loss(t_patience.float(), st_patience.float())+F.mse_loss(t_attn.float(),s_attn.float()))/2

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    "compute the label smoothed nll loss"
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(0) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@register_criterion(
    "pet_warm_up_criterion", dataclass=PETLabelSmoothedCrossEntropyCriterionConfig
)
class PTPCrossEntropyCriterionUpdate(FairseqCriterion):
    "custom loss for warming up the PET model"
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,

            ignore_prefix_size=0,
            report_accuracy=False,

            alpha=0.1, 
            beta=100,
            T=10,

            # for ptp label
            confidence_score_t=0.9,  # 0.6
            confidence_score_f=0.5,  # 0.6

    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.alpha = alpha
        self.beta = beta
        self.T = T

        self.label_size = 4
        self.confidence_score_t = confidence_score_t
        self.confidence_score_f = confidence_score_f

        self.label_statistic = {}
        self.init_label_statistic()

    def init_label_statistic(self):
        for i in range(self.label_size):
            self.label_statistic[str(i)] = 0

    def get_label_statistic(self):
        print(f"[Label Dictionary]\n {self.label_statistic}")

    def get_lprobs_and_target_kd(self, model, net_output, sample, lprobs=True, for_generate_label=False):

        lprobs = model.get_normalized_probs_kd(net_output, log_probs=lprobs)  # output에 log_softmax 취한 값
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        if for_generate_label == True:
            return lprobs, target.view(-1)
        else:
            return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def generate_ptp_label(self, model, net_output, sample):

        # teacher model이 sample data에 대해 forward한 후 log_softmax
        probs = model.get_normalized_probs_kd(net_output, log_probs=False)  # output에 softmax 취한 값
        target = model.get_targets(sample, net_output)

        if self.ignore_prefix_size > 0:
            # probs: B x T x C (C: size of decoder's embedding)
            probs = probs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()

        batch_sz, t_sz, class_sz = probs.size()
        pred_logit, pred_target = torch.max(probs, 2)
        ptp_label = torch.zeros([batch_sz, t_sz])
        separators_t = [0.9]
        separators_f = [0.7]
        # generate ptp label
        for batch_idx in range(batch_sz):
            for t_idx in range(t_sz):
                flag = False
                token_pred_logit = pred_logit[batch_idx][t_idx]
                if pred_target[batch_idx][t_idx] == target[batch_idx][t_idx]:  # 정답
                    for idx, sep in enumerate(separators_t):
                        if token_pred_logit > sep:
                            self.label_statistic[f"{idx}"] += 1
                            ptp_label[batch_idx][t_idx] = idx
                            flag = True
                            break
                        else:
                            continue
                    if flag == False:
                        self.label_statistic[f"{int(self.label_size / 2) - 1}"] += 1
                        ptp_label[batch_idx][t_idx] = idx
                else:
                    for idx, sep in enumerate(separators_f):
                        if token_pred_logit > sep:
                            self.label_statistic[f"{int(self.label_size / 2) + idx}"] += 1
                            ptp_label[batch_idx][t_idx] = idx
                            break
                        else:
                            continue
                    if flag == False:
                        self.label_statistic[f"{self.label_size - 1}"] += 1
                        ptp_label[batch_idx][t_idx] = idx

        ptp_label = ptp_label.long()

        return ptp_label.to('cuda:0')

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target_kd(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        return loss, nll_loss

    def get_ptp_output(self, model, net_output, sample):
        """convert model's output info ptp prediction form
            P(correct) = probs[:,:,true_label] #BxT
            P(incorrect) = 1-P(correct)
            P(convince | correct) = P(convince | correct) = max(probs)       #BxT
            P(not_convince | correct) = P(not_convince | correct) = 1-max(probs)

            ** Arguments **
            model: student model
            net_output: [batch, #tokens, #class]
            sample: input data
        """
        probs, ori_target = self.get_lprobs_and_target_kd(model, net_output, sample, lprobs=False,
                                                          for_generate_label=True)
        pred_logit, pred_target = torch.max(probs, 2)  #BxT, BxT

        # P(correct), P(wrong)
        ori_target=ori_target.view(pred_target.size()[0],-1) #BxT
        if ori_target.dim() == probs.dim() - 1:
            ori_target = ori_target.unsqueeze(-1)  #BxTx1
        p_corr = probs.gather(dim=-1, index=ori_target) #BxTx1
        p_wrng = 1-p_corr
        
        # P(confident | {correct, wrong})
        p_conv = pred_logit.unsqueeze(-1) #BxTx1
        p_unconv = 1-p_conv

        # P({confident, not confident} & {correct, wrong})
        p_tt, p_t = p_conv * p_corr, p_unconv * p_corr #BxTx1
        p_ff, p_f = p_conv * p_wrng, p_unconv * p_wrng #BxTx1

        # Final Prob. dist. for PTP label
        ptp_probs = torch.cat([p_tt, p_t, p_ff, p_f], dim = -1)
        return torch.log(ptp_probs)

    def compute_ptp_loss(self, teacher_model, teacher_net_output, model, net_output, sample, reduce=True):
        # generate ptp label
        ptp_target = self.generate_ptp_label(teacher_model, teacher_net_output, sample)  # target size: batch_size
        # forward with ptp label
        ptp_lprobs = self.get_ptp_output(model, net_output, sample)
        # compute CE Loss
        loss, nll_loss = label_smoothed_nll_loss(
            ptp_lprobs,
            ptp_target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss  

    def forward(self, model, teacher_model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        if teacher_model is not None:
            with torch.no_grad():
                teacher_net_output, teacher_net_output_en = teacher_model(**sample["net_input"])
                net_output, net_output_en = model(**sample["net_input"])
                loss, nll_loss = self.compute_ptp_loss(teacher_model, teacher_net_output, model, net_output, sample,
                                                       reduce=reduce)
            #self.get_label_statistic()

        else:
            net_output, net_output_en = model(**sample["net_input"])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion(
    "pet_cross_entropy_adjust", dataclass=PETLabelSmoothedCrossEntropyCriterionConfig
)
class PTPCrossEntropyCriterion_adjust(FairseqCriterion):
    "variation of pet_warm_up_criterion"
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,

        ignore_prefix_size=0,
        report_accuracy=False,

        alpha=0.1, 
        beta=100,
        T=10,

        # for ptp label
        confidence_score_t = 0.9, #0.6
        confidence_score_f = 0.5, #0.6

    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.alpha=alpha
        self.beta=beta
        self.T=T

        self.label_size=6
        self.confidence_score_t = confidence_score_t
        self.confidence_score_f = confidence_score_f

        self.label_statistic={}
        self.init_label_statistic()

    def init_label_statistic(self):
        for i in range(self.label_size):
            self.label_statistic[str(i)]=0
    def get_label_statistic(self):
        print(f"[Label Dictionary]\n {self.label_statistic}")


    def get_lprobs_and_target_kd(self, model, net_output, sample, lprobs=True, for_generate_label = False):

        lprobs = model.get_normalized_probs_kd(net_output, log_probs=lprobs)  # output에 log_softmax 취한 값
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        if for_generate_label==True:
            return lprobs, target.view(-1)
        else:
            return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def generate_ptp_label(self, model, net_output, sample):

        #teacher model이 sample data에 대해 forward한 후 log_softmax
        probs = model.get_normalized_probs_kd(net_output, log_probs=False) #output에 softmax 취한 값
        target = model.get_targets(sample, net_output)

        if self.ignore_prefix_size > 0:
            # probs: B x T x C (C: size of decoder's embedding)
            probs = probs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()

        batch_sz, t_sz, class_sz = probs.size()
        pred_logit, pred_target = torch.max(probs,2)
        ptp_label = torch.zeros([batch_sz, t_sz])
        separators_t = [0.9,0.0]
        separators_f = [0.5,0.05]
        #generate ptp label
        for batch_idx in range(batch_sz):
            for t_idx in range(t_sz):
                flag = False
                token_pred_logit=pred_logit[batch_idx][t_idx]
                if pred_target[batch_idx][t_idx] == target[batch_idx][t_idx]: #정답
                    for idx, sep in enumerate(separators_t):
                        if token_pred_logit > sep:
                            self.label_statistic[f"{idx}"]+=1
                            ptp_label[batch_idx][t_idx]=idx
                            flag = True
                            break
                        else:
                            continue
                    if flag == False:
                        self.label_statistic[f"{int(self.label_size/2)-1}"]+=1
                        ptp_label[batch_idx][t_idx]=idx
                else:
                    for idx, sep in enumerate(separators_f):
                        if token_pred_logit > sep:
                            self.label_statistic[f"{int(self.label_size/2)+idx}"]+=1
                            ptp_label[batch_idx][t_idx]=idx
                            break
                        else:
                            continue
                    if flag == False:
                        self.label_statistic[f"{self.label_size-1}"]+=1
                        ptp_label[batch_idx][t_idx]=idx

        ptp_label = ptp_label.long()

        return ptp_label.to('cuda:0')

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target_kd(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        return loss, nll_loss
    def get_ptp_output(self, model, net_output, sample):
        """convert model's output info ptp prediction form
             model: student model
             net_output: [batch, #tokens, #class]
             sample: input data
        """
        probs, true_label = self.get_lprobs_and_target_kd(model, net_output, sample, lprobs=False, for_generate_label=True)
        pred_logit, pred_target = torch.max(probs, 2) #batch, tokens

        #dist = compute confidence score
        dist_value_list=list(self.label_statistic.values()) #teacher's dist to ptp label
        true_dist = torch.Tensor(dist_value_list[:int(self.label_size/2)])
        true_dist = true_dist / torch.sum(true_dist)
        false_dist = torch.Tensor(dist_value_list[int(self.label_size/2):])
        false_dist = false_dist / torch.sum(false_dist)
        dist = torch.cat([true_dist, false_dist]).to('cuda:0')

        # pred_logit form: [P(True),...,P(True), P(False),...,P(False)]
        pred_logit=torch.stack([pred_logit for _ in range(int(self.label_size/2))], dim=2)
        pred_logit=torch.cat([pred_logit, 1-pred_logit],dim=2)
        ptp_lprobs=pred_logit*dist
        print("dist",dist)
        print("pred_logit",pred_logit[0][0])
        print("ptp_lprobs",ptp_lprobs[0][0])
        print("pred_logit",pred_logit[1][1])
        print("ptp_lprobs",ptp_lprobs[1][1])
        return torch.log(ptp_lprobs)
    
    def get_ptp_output2(self, model, net_output, sample):
        """convert model's output info ptp prediction form
            => this version doesn't use the dist of teacher to ptp label.
                Instead, it use a heuristically pre-defined dist.
             model: student model
             net_output: [batch, #tokens, #class]
             sample: input data
        """
        probs, true_label = self.get_lprobs_and_target_kd(model, net_output, sample, lprobs=False, for_generate_label=True)
        pred_logit, pred_target = torch.max(probs, 2) #batch, tokens

        score = [0.99, 0.099, 0.001, 5e-6, 0.000095, 0.9999]
        dist = torch.Tensor(score).to('cuda:0') #pre-defined dist to ptp label

        # pred_logit form: [P(True),...,P(True), P(False),...,P(False)]
        pred_logit=torch.stack([pred_logit for _ in range(int(self.label_size/2))], dim=2)
        pred_logit=torch.cat([pred_logit, 1-pred_logit],dim=2)
        ptp_lprobs=pred_logit*dist
        return torch.log(ptp_lprobs)

    def compute_ptp_loss(self, teacher_model, teacher_net_output, model, net_output, sample, reduce=True):
        # generate ptp label
        ptp_target = self.generate_ptp_label(teacher_model, teacher_net_output,sample) #target size: batch_size

        # forward with ptp label
        ptp_lprobs = self.get_ptp_output2(model,net_output,sample)
        # compute CE Loss
        loss, nll_loss = label_smoothed_nll_loss(
            ptp_lprobs,
            ptp_target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss #이건 CE

    def forward(self, model, teacher_model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        #net_output, net_output_en = model(**sample["net_input"])

        if teacher_model is not None:
            with torch.no_grad():
                teacher_net_output, teacher_net_output_en=teacher_model(**sample["net_input"])
                net_output, net_output_en = model(**sample["net_input"])
                loss, nll_loss = self.compute_ptp_loss(teacher_model,teacher_net_output, model, net_output, sample, reduce=reduce)
                #self.get_label_statistic()

        else:
            net_output, net_output_en = model(**sample["net_input"])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output



    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion(
    "pet_kd_criterion", dataclass=PETLabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion_kd(FairseqCriterion):
    "loss for train the pet model using vanilla knowledge distillation"
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,

        alpha=0.05,
        beta=100,
        T=10,

        ignore_prefix_size=0,
        report_accuracy=False

    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

        self.alpha=alpha
        self.beta=beta
        self.T=T

    def forward(self, model, teacher_model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output,_= model(**sample["net_input"])
        if teacher_model is not None:
            with torch.no_grad():
                teacher_net_output,_=teacher_model(**sample["net_input"])

            loss, nll_loss = self.compute_loss_kd(model, net_output, teacher_net_output, sample, reduce=reduce)
        else:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target_kd(self, model, net_output, sample):
        lprobs = model.get_normalized_probs_kd(net_output, log_probs=True) #output에 log_softmax 취한 값
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target_kd(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        return loss, nll_loss

    def compute_loss_kd(self, model, net_output, teacher_net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target_kd(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        d_loss = distillation_loss_kd(net_output, teacher_net_output, self.T)
        p_loss=0.0
        loss= loss*(1-self.alpha)+d_loss*(self.alpha)+p_loss*(self.beta)
        return loss, nll_loss #이건 CE

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

@register_criterion(
    "pet_pkd_criterion", dataclass=PETLabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion_pkd_wo_torch(FairseqCriterion):
    "loss for train the pet model using the patient knowledge distillation using the customized functions"
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,

        ignore_prefix_size=0,
        report_accuracy=False,

        alpha = 0.1,
        beta = 50,
        T = 10,

        distill_type = "skip"
        #α between {0.1 and 0.7}, and β between {0 and 500}
        # 1-alpha: ce loss, alpha: distill loss, beta: pkd loss

    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        ### edited by hyojin
        self.alpha=alpha
        self.beta=beta
        self.T=T
        self.distill_type = distill_type

    def get_cond(self):
        return [self.alpha, self.beta, self.T]

    def forward(self, model, teacher_model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output, net_output_en = model(**sample["net_input"])
        """
        1. net_output (Tuple): decoder's output
            a. net_output[0]-(Tensor): output from the final layer (will be converted to logit)
                (final output states: [batch, tgt_len, embed_dim])
            b. net_output[1]-(Dicitonary): dictionary of extra information
                i.  net_output[1]["attn"]-(Tensor): attention weights from the final layer
                ii. net_output[1]["inner_states"]-(List[Tensor]): List of hidden states from the interm. layers
                    (hidden_states: [batch, tgt_len, embed_dim]) 
        2. net_output_en (dict): encoder's output
            a. net_output_en["encoder_out"]-(Tensor): output from the final layer
                (final output states: [src_len, batch, embed_dim])
            b. net_output_en["encoder_states"]-(List[Tensor]): List of hidden states from the interm. layers
                (hidden states: [src_len, batch, embed_dim] // if return_all_hiddens == True )
        """
        if teacher_model is not None:
            with torch.no_grad():
                teacher_net_output, teacher_net_output_en=teacher_model(**sample["net_input"])
            loss, nll_loss = self.compute_loss_pkd(model, net_output,net_output_en, teacher_net_output,teacher_net_output_en, sample, reduce=reduce)
        else:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target_kd(self, model, net_output, sample):

        lprobs = model.get_normalized_probs_kd(net_output, log_probs=True) #output에 log_softmax 취한 값
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target_kd(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        return loss, nll_loss

    def compute_loss_pkd(self, model, net_output,net_output_en,teacher_net_output, teacher_net_output_en, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target_kd(model, net_output, sample)
        """
        L_{pkd} = ((1-alpha) * L_CE) + (alpha * L_DS) + (beta * L_PT)
        """
        # 1. L_CE of student
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        # 2. L_DS & L_PT
        d_loss = distillation_loss_kd_no_torch(net_output, teacher_net_output, self.T)
        p_loss = patience_loss_kd_no_torch(net_output,net_output_en,teacher_net_output,teacher_net_output_en,
                                           normalized_patience=True, distill_type=self.distill_type)
        loss= loss*(1-self.alpha)+d_loss*(self.alpha)+p_loss*(self.beta)
        return loss, nll_loss #이건 CE

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
