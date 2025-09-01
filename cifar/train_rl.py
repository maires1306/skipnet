""" This file for training SkipNet in Hybrid RL stage + Manual Gating options.
Support PyTorch 2.0 and single GPU only.
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import time
import logging

import models
from data import *
import math

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))


# --------------------------- Loss helpers ---------------------------

class BatchCrossEntropy(nn.Module):
    def __init__(self):
        super(BatchCrossEntropy, self).__init__()

    def forward(self, x, target):
        # target deve ser long (B,)
        target = target.long().view(-1, 1)
        logp = F.log_softmax(x, dim=1)
        output = -logp.gather(1, target)  # (B,1)
        return output


# --------------------------- Manual policy (RNN) ---------------------------

class StaticRNNPolicy(nn.Module):
    """
    Política estática para substituir o controlador RNN durante o forward.
    Devolve máscaras (0=pula, 1=executa) seguindo uma sequência fixa.
    Compatível com a interface esperada por ResNetRecurrentGateRL:
      - tem .hidden, .saved_actions, .rewards
      - expõe .init_hidden e .repackage_hidden()
      - forward(x) -> (mask(B,1,1,1), bi_prob(B,2))
    """
    def __init__(self, seq_values, default_value=1.0):
        super().__init__()
        self.seq = [float(v) for v in seq_values]
        self.default = float(default_value)
        self.idx = 0
        self.hidden = None
        self.saved_actions = []  # mantemos a API
        self.rewards = []

    def init_hidden(self, batch_size):
        self.hidden = None
        return None

    def repackage_hidden(self):
        return

    def forward(self, x):
        B = x.size(0)
        if self.idx < len(self.seq):
            v = self.seq[self.idx]
        else:
            v = self.default
        self.idx += 1
        mask = torch.full((B, 1, 1, 1), v, device=x.device, dtype=x.dtype)
        # bi_prob = [p(skip)=1-mask, p(exec)=mask] apenas para logging/compat
        bi_prob = torch.cat([1.0 - mask.view(B, 1), mask.view(B, 1)], dim=1)
        return mask, bi_prob


def estimate_num_gates(model):
    """
    Para ResNetRecurrentGateRL, #gates = (sum(layers) - 1).
    Ex.: [6,6,6] -> 18 blocos, 17 gates (não há gate no último bloco).
    Se não existir atributo, usamos 17 como fallback (ResNet-38).
    """
    if hasattr(model, "num_layers"):
        try:
            total_blocks = sum(model.num_layers)
            return max(1, total_blocks - 1)
        except Exception:
            pass
    return 17  # fallback razoável para ResNet-38


def build_manual_sequence(args, num_gates):
    mode = args.manual_gate_mode
    if mode == "all_exec":
        return [1] * num_gates
    if mode == "all_skip":
        return [0] * num_gates
    if mode == "list":
        if not args.manual_gate_list:
            raise ValueError("--manual-gate-list vazio para modo 'list'")
        s = args.manual_gate_list.strip()
        # aceita "1,0,1,..." ou "10110"
        if "," in s:
            vals = [int(x) for x in s.split(",") if x.strip() != ""]
        else:
            vals = [int(ch) for ch in s if ch in ("0", "1")]
        if len(vals) == 0:
            raise ValueError("Não consegui parsear --manual-gate-list")
        # se lista for menor, repete o último valor
        if len(vals) < num_gates:
            vals = vals + [vals[-1]] * (num_gates - len(vals))
        else:
            vals = vals[:num_gates]
        return vals
    # none
    return None


def maybe_install_manual_gating(args, model):
    """
    Se manual gating estiver ativo e gate-type == rnn:
      - substitui model.control por StaticRNNPolicy com a sequência pedida.
      - retorna (manual_active=True)
    Para gate-type ff: apenas emite aviso.
    """
    manual_active = (args.manual_gate_mode != "none")
    if not manual_active:
        return False

    if args.gate_type == "rnn":
        num_gates = estimate_num_gates(model)
        seq = build_manual_sequence(args, num_gates)
        if seq is None:
            # modo none (não deveria cair aqui)
            return False
        # instala política estática
        model.control = StaticRNNPolicy(seq, default_value=seq[-1])
        # Em modo manual, os termos de RL não fazem sentido:
        args.alpha = 0.0
        args.rl_weight = 0.0
        logging.info(f"[Manual Gating] Ativado ({args.manual_gate_mode}); "
                     f"{num_gates} gates; alpha=0, rl-weight=0")
        return True
    else:
        logging.warning("[Manual Gating] gate-type=ff ainda não implementado "
                        "para override direto no train_rl. "
                        "Use --gate-type rnn para estas opções.")
        return False


# --------------------------- Argparse ---------------------------

def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR training with SkipNet HRL + Manual Gating')
    parser.add_argument('cmd', choices=['train', 'test', 'tune'])
    parser.add_argument('arch', metavar='ARCH',
                        default='cifar10_rnn_gate_rl_38',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: cifar10_rnn_gate_rl_38)')
    parser.add_argument('--gate-type', default='rnn', choices=['ff', 'rnn'],
                        help='gate type')
    parser.add_argument('--dataset', '-d', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn'],
                        help='dataset type')
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--iters', default=10000, type=int,
                        help='total iterations')
    parser.add_argument('--start-iter', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency')
    parser.add_argument('--resume', default='', type=str,
                        help='path to checkpoint')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='lr step ratio')
    parser.add_argument('--warm-up', action='store_true',
                        help='warm up for n=18 (first 400 iters)')
    parser.add_argument('--save-folder', default='save_checkpoints',
                        type=str, help='checkpoints folder')
    parser.add_argument('--eval-every', default=200, type=int,
                        help='evaluate every N iterations')
    parser.add_argument('--fine_tune', action='store_true',
                        help='fine tune model')

    # -------- RL params --------
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='reward magnitude for avg # skipped layers')
    parser.add_argument('--temperature', type=float, default=1,
                        help='softmax temperature')
    parser.add_argument('--rl-weight', default=0.01, type=float,
                        help='rl weight (policy gradient)')
    parser.add_argument('--gamma', default=1, type=float,
                        help='discount factor')
    parser.add_argument('--restart', action='store_true',
                        help='restart training')

    # -------- Manual gating --------
    parser.add_argument('--manual-gate-mode',
                        choices=['none', 'all_exec', 'all_skip', 'list'],
                        default='none',
                        help='override de gating sem RL')
    parser.add_argument('--manual-gate-list', type=str, default='',
                        help="lista de 0/1 (ex: '1,1,0,...' ou '111000')")
    parser.add_argument('--freeze-gates', action='store_true',
                        help='congela parâmetros do controlador (sem RL)')

    args = parser.parse_args()
    return args


# --------------------------- Main / Train / Eval ---------------------------

def main():
    args = parse_args()
    save_path = args.save_path = os.path.join(args.save_folder, args.arch)
    os.makedirs(save_path, exist_ok=True)

    # logger
    args.logger_file = os.path.join(save_path, f'log_{args.cmd}.txt')
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)
    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)
    elif args.cmd == 'tune':
        import ray
        import ray.tune as tune
        from ray.tune import Experiment
        from ray.tune.median_stopping_rule import MedianStoppingRule
        ray.init()
        sched = MedianStoppingRule(
            time_attr="timesteps_total", reward_attr="neg_mean_loss")
        tune.register_trainable(
            "run_training", lambda cfg, reporter: run_training(args, cfg, reporter))
        experiment = Experiment("train_rl", "run_training", trial_resources={"gpu": 1},
                                config={"alpha": tune.grid_search([0.1, 0.01, 0.001])})
        tune.run_experiments(experiment, scheduler=sched, verbose=False)


def run_training(args, tune_config={}, reporter=None):
    vars(args).update(tune_config)

    # create model
    model = models.__dict__[args.arch](args.pretrained).cuda()

    # extract gate actions and rewards (handles both ff/rnn names)
    if args.gate_type == 'ff':
        # Em RL FF original: instâncias estão em model.gate_instances (ver models.py)
        gate_saved_actions = getattr(model, 'saved_actions', [])
        gate_rewards = getattr(model, 'rewards', [])
    elif args.gate_type == 'rnn':
        gate_saved_actions = model.control.saved_actions
        gate_rewards = model.control.rewards
    else:
        raise ValueError("gate-type inválido")

    # instalar override manual, se pedido
    manual_active = maybe_install_manual_gating(args, model)

    # opção de congelar controlador (sem RL), mas mantendo gating aprendido
    if args.freeze_gates and args.gate_type == 'rnn' and not manual_active:
        for p in model.control.parameters():
            p.requires_grad = False
        # também desativa RL
        args.alpha = 0.0
        args.rl_weight = 0.0
        logging.info("[Freeze Gates] controlador congelado; alpha=0, rl-weight=0")

    best_prec1 = 0

    # carregar checkpoint (SP ou HRL)
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda')
            if args.restart:
                best_prec1 = checkpoint.get('best_prec1', 0.0)
                args.start_iter = checkpoint.get('iter', 0)
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint.get('iter', 'n/a')))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = True

    train_loader = prepare_train_data(dataset=args.dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    # losses e otimização
    criterion = BatchCrossEntropy().cuda()
    total_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_rewards = AverageMeter()
    total_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()

    end = time.time()

    # cada batch é um "episódio"
    print('start: ', args.start_iter)
    for i in range(args.start_iter, args.iters):
        model.train()
        adjust_learning_rate(args, optimizer, i)

        input, target = next(iter(train_loader))
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = input.cuda(non_blocking=True)
        target_var = target

        # forward
        output, masks, probs = model(input_var)

        # coleta de skip ratio por gate
        skips = [mask.detach().le(0.5).float().mean().item() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))

        # perdas
        pred_loss = criterion(output, target_var)  # (B,1)

        # ---------- RL apenas se NÃO estiver em modo manual/freeze ----------
        if (args.rl_weight > 0) and (args.alpha != 0) and (len(gate_saved_actions) > 0):
            normalized_alpha = args.alpha / max(1, len(gate_saved_actions))
            for act in gate_saved_actions:
                gate_rewards.append((1 - act.float()).detach() * normalized_alpha)

            # retornos cumulativos (usa -pred_loss como baseline)
            R = -pred_loss.detach()  # (B,1)
            cum_rewards = []
            for r in gate_rewards[::-1]:
                R = r + args.gamma * R
                cum_rewards.insert(0, R)

            # policy gradient (probabilidades em `probs`)
            policy_terms = []
            # `probs` é uma lista de distribuições por gate (B,2) para rnn; em ff pode ser (B,2)
            for action, reward, p in zip(gate_saved_actions, cum_rewards, probs):
                logp = torch.log(torch.clamp(p, min=1e-8)).gather(1, action.view(-1, 1).long())  # (B,1)
                policy_terms.append((reward * logp).mean())

            policy_loss = -sum(policy_terms) if len(policy_terms) > 0 else torch.zeros((), device=output.device)
            total_loss = total_criterion(output, target_var) + args.rl_weight * policy_loss
        else:
            # Sem RL (manual/freeze ou sem ações salvas): treina só CE
            total_loss = total_criterion(output, target_var)
            cum_rewards = []
            policy_loss = torch.zeros((), device=output.device)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # métricas
        prec1, = accuracy(output, target, topk=(1,))
        total_rewards.update(cum_rewards[0].mean().item() if len(cum_rewards) > 0 else 0.0, input.size(0))
        total_losses.update(total_loss.mean().item(), input.size(0))
        losses.update(pred_loss.mean().item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        skip_ratios.update(skips, input.size(0))
        total_gate_reward = float(sum([r.mean().item() for r in gate_rewards])) if len(gate_rewards) > 0 else 0.0

        # limpar buffers de RL
        if isinstance(gate_saved_actions, list):
            del gate_saved_actions[:]
        if isinstance(gate_rewards, list):
            del gate_rewards[:]

        batch_time.update(time.time() - end)
        end = time.time()

        if reporter:
            reporter(timesteps_total=i, neg_mean_loss=losses.val)

        if i % args.print_freq == 0 or i == (args.iters - 1):
            logging.info(
                "Iter: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Total reward {total_rewards.val: .3f}({total_rewards.avg: .3f})\t"
                "Total gate reward {total_gate_reward: .3f}\t"
                "Total Loss {total_losses.val:.3f} ({total_losses.avg:.3f})\t"
                "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, args.iters,
                    batch_time=batch_time,
                    data_time=data_time,
                    total_rewards=total_rewards,
                    total_gate_reward=total_gate_reward,
                    total_losses=total_losses,
                    loss=losses,
                    top1=top1)
            )

        # avaliação
        if (i % args.eval_every == 0) or (i == (args.iters - 1)):
            prec1, cp = validate(args, test_loader, model)

            # limpar buffers de RL (por segurança)
            if isinstance(gate_saved_actions, list):
                del gate_saved_actions[:]
            if isinstance(gate_rewards, list):
                del gate_rewards[:]

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            checkpoint_path = os.path.join(args.save_path, f'checkpoint_{i:05d}.pth.tar')
            save_checkpoint({
                'iter': i,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=checkpoint_path)
            shutil.copyfile(checkpoint_path, os.path.join(args.save_path, 'checkpoint_latest.pth.tar'))


def validate(args, test_loader, model):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            output, masks, probs = model(input.cuda(non_blocking=True))

        skips = [mask.detach().le(0.5).float().mean().item() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))

        prec1, = accuracy(output, target, topk=(1,))
        top1.update(prec1.item(), input.size(0))
        skip_ratios.update(skips, input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or (i == (len(test_loader) - 1)):
            logging.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time, top1=top1
                )
            )
    logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    skip_summaries = [1 - skip_ratios.avg[idx] for idx in range(skip_ratios.len)]
    cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100.0
    logging.info('*** Computation Percentage: {:.3f} %'.format(cp))

    return top1.avg, cp


def test_model(args):
    model = models.__dict__[args.arch](args.pretrained).cuda()

    # permitir teste com modo manual também
    _ = maybe_install_manual_gating(args, model)

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda')
            args.start_iter = checkpoint.get('iter', 0)
            best_prec1 = checkpoint.get('best_prec1', 0.0)
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint.get('iter', 'n/a')))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    validate(args, test_loader, model)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path, 'model_best.pth.tar'))


# --------------------------- meters & utils ---------------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # upper bound
        self.reset()
    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0
    def set_len(self, n):
        self.len = n
        self.reset()
    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count


def adjust_learning_rate(args, optimizer, _iter):
    """ divide lr por 10 em 40k e 60k (mantido para compat) """
    if args.warm_up and (_iter < 400):
        lr = 0.01
    elif 40000 <= _iter < 60000:
        lr = args.lr * (args.step_ratio ** 1)
    elif _iter >= 60000:
        lr = args.lr * (args.step_ratio ** 2)
    else:
        lr = args.lr

    if _iter % args.eval_every == 0:
        logging.info('Iter [{}] learning rate = {}'.format(_iter, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
