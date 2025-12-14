"""
FrEVL Optimizer and Learning Rate Scheduler Utilities
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import warnings

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


# ============================================================================
# Custom Optimizers
# ============================================================================

class LAMB(Optimizer):
    """
    LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
    Reference: https://arxiv.org/abs/1904.00962
    
    Particularly effective for large batch training
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        adam: bool = False,
        bias_correction: bool = True
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            adam=adam,
            bias_correction=bias_correction
        )
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                if group['bias_correction']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    exp_avg_hat = exp_avg / bias_correction1
                    exp_avg_sq_hat = exp_avg_sq / bias_correction2
                else:
                    exp_avg_hat = exp_avg
                    exp_avg_sq_hat = exp_avg_sq
                
                # Adam update
                update = exp_avg_hat / (exp_avg_sq_hat.sqrt() + group['eps'])
                
                # Weight decay
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])
                
                # Layer adaptation
                if group['adam']:
                    # Use Adam update directly
                    p.data.add_(update, alpha=-group['lr'])
                else:
                    # LAMB layer adaptation
                    w_norm = p.data.pow(2).sum().sqrt()
                    g_norm = update.pow(2).sum().sqrt()
                    
                    # Compute adaptive learning rate
                    trust_ratio = 1.0
                    if w_norm > 0 and g_norm > 0:
                        trust_ratio = w_norm / g_norm
                    
                    p.data.add_(update, alpha=-group['lr'] * trust_ratio)
        
        return loss


class Lookahead(Optimizer):
    """
    Lookahead optimizer
    Reference: https://arxiv.org/abs/1907.08610
    
    Wrapper that implements Lookahead on any base optimizer
    """
    
    def __init__(self, base_optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if k < 1:
            raise ValueError(f"Invalid k: {k}")
        
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        self.defaults = base_optimizer.defaults
        
        # Store slow weights
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['slow_weight'] = p.data.clone()
    
    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        
        self.step_count += 1
        if self.step_count % self.k == 0:
            # Update slow weights
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if 'slow_weight' in state:
                        slow = state['slow_weight']
                        fast = p.data
                        slow.add_(fast - slow, alpha=self.alpha)
                        p.data.copy_(slow)
        
        return loss
    
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups
    
    @param_groups.setter
    def param_groups(self, value):
        self.base_optimizer.param_groups = value


# ============================================================================
# Learning Rate Schedulers
# ============================================================================

class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine learning rate schedule with linear warmup
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class PolynomialLRScheduler(_LRScheduler):
    """
    Polynomial learning rate schedule with warmup
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        power: float = 1.0,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Polynomial decay
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            decay = (1 - progress) ** self.power
            return [
                self.min_lr + (base_lr - self.min_lr) * decay
                for base_lr in self.base_lrs
            ]


class CyclicCosineScheduler(_LRScheduler):
    """
    Cyclic cosine annealing schedule
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        cycle_steps: int,
        min_lr: float = 0,
        warmup_steps: int = 0,
        last_epoch: int = -1
    ):
        self.cycle_steps = cycle_steps
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cyclic cosine
            cycle_pos = (self.last_epoch - self.warmup_steps) % self.cycle_steps
            progress = cycle_pos / self.cycle_steps
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


# ============================================================================
# Factory Functions
# ============================================================================

def create_optimizer(
    model: torch.nn.Module,
    config: Any,
    filter_frozen: bool = True
) -> Optimizer:
    """
    Create optimizer with parameter groups
    
    Args:
        model: Model to optimize
        config: Configuration object with optimizer settings
        filter_frozen: Whether to filter frozen parameters
    
    Returns:
        Optimizer instance
    """
    
    # Filter parameters
    if filter_frozen:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.parameters()
    
    # Group parameters by weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight', 'norm']
    
    param_groups = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            'weight_decay': config.weight_decay
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            'weight_decay': 0.0
        }
    ]
    
    # Remove empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    # Create optimizer
    optimizer_name = config.optimizer.lower()
    
    if optimizer_name == 'adamw':
        from torch.optim import AdamW
        optimizer = AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=config.learning_rate,
            momentum=0.9,
            nesterov=True
        )
    elif optimizer_name == 'lamb':
        optimizer = LAMB(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay
        )
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=config.learning_rate,
            alpha=0.99,
            eps=1e-8,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    # Apply Lookahead if requested
    if hasattr(config, 'use_lookahead') and config.use_lookahead:
        optimizer = Lookahead(
            optimizer,
            k=getattr(config, 'lookahead_k', 5),
            alpha=getattr(config, 'lookahead_alpha', 0.5)
        )
    
    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    config: Any,
    steps_per_epoch: int
) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration object with scheduler settings
        steps_per_epoch: Number of steps per epoch
    
    Returns:
        Scheduler instance or None
    """
    
    if not hasattr(config, 'scheduler') or config.scheduler is None:
        return None
    
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio) if hasattr(config, 'warmup_ratio') else 0
    
    scheduler_name = config.scheduler.lower()
    
    if scheduler_name == 'cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=config.learning_rate * 0.01
        )
    elif scheduler_name == 'linear':
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps if warmup_steps > 0 else total_steps
        )
    elif scheduler_name == 'exponential':
        from torch.optim.lr_scheduler import ExponentialLR
        scheduler = ExponentialLR(
            optimizer,
            gamma=0.95
        )
    elif scheduler_name == 'polynomial':
        scheduler = PolynomialLRScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            power=1.0,
            min_lr=config.learning_rate * 0.01
        )
    elif scheduler_name == 'onecycle':
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_ratio if hasattr(config, 'warmup_ratio') else 0.1,
            anneal_strategy='cos'
        )
    elif scheduler_name == 'cyclic':
        scheduler = CyclicCosineScheduler(
            optimizer,
            cycle_steps=steps_per_epoch * 2,  # 2 epochs per cycle
            min_lr=config.learning_rate * 0.01,
            warmup_steps=warmup_steps
        )
    elif scheduler_name == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=config.learning_rate * 0.001
        )
    else:
        warnings.warn(f"Unknown scheduler: {config.scheduler}, using no scheduler")
        scheduler = None
    
    return scheduler


def get_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float = 0.01,
    learning_rate_scale: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """
    Get parameter groups with different learning rates and weight decay
    
    Args:
        model: Model to get parameters from
        weight_decay: Default weight decay
        learning_rate_scale: Dictionary mapping parameter name patterns to LR scales
    
    Returns:
        List of parameter groups
    """
    
    if learning_rate_scale is None:
        learning_rate_scale = {}
    
    # Default groups
    no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight', 'norm']
    
    # Organize parameters
    grouped_parameters = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Determine learning rate scale
        lr_scale = 1.0
        for pattern, scale in learning_rate_scale.items():
            if pattern in name:
                lr_scale = scale
                break
        
        # Determine weight decay
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        
        # Create group key
        group_key = (lr_scale, wd)
        
        if group_key not in grouped_parameters:
            grouped_parameters[group_key] = []
        
        grouped_parameters[group_key].append(param)
    
    # Create parameter groups
    param_groups = []
    for (lr_scale, wd), params in grouped_parameters.items():
        param_groups.append({
            'params': params,
            'lr_scale': lr_scale,
            'weight_decay': wd
        })
    
    return param_groups


# ============================================================================
# Learning Rate Utilities
# ============================================================================

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    """
    Create a cosine learning rate schedule with warmup
    Compatible with HuggingFace Transformers
    """
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
):
    """
    Create a linear learning rate schedule with warmup
    Compatible with HuggingFace Transformers
    """
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda, last_epoch)
