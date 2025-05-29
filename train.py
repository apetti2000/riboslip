import datetime
import itertools
import os

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import tqdm

import data_loading


def main_loop(model_fns,
              data,
              device,
              base_name,
              lrs,
              wds,
              dataset_splits,
              schedulers,
              epochs: int=100,
              baselines=None,
              loss_mixin: float=0.5,
              save_path: os.PathLike=None,
              data_augment: bool=False):
  metrics = []

  if baselines:
    for mix in baselines:
      if isinstance(mix, tuple):
        mix_name = f'{mix[0]}_to_{mix[1]}'
      else:
        mix_name = mix
      writer = SummaryWriter(f'logs/{base_name}/baseline_{mix_name}')
      for k, v in baselines[mix_name].items():
        for i in range(epochs):
          writer.add_scalar(k, v, i)


  for lr, wd, scheduler_name, mix, model_fn in itertools.product(lrs, wds, schedulers, dataset_splits, model_fns):
    scheduler_dict = {
      'StepLR': lambda opt: optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1),
      'ExponentialLR': lambda opt: optim.lr_scheduler.ExponentialLR(opt, gamma=0.95),
      'CosineAnnealingLR': lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50),
      'CyclicLR': lambda opt: optim.lr_scheduler.CyclicLR(opt, base_lr=lr * 0.1, max_lr=lr * 1.5, step_size_up=30, mode='triangular')
    }

    train_loader, val_loader = data_loading.build_dataloaders(data, 'minus', test_split=mix, data_augment=data_augment)
    model = model_fn().to(device)
    classification_criterion = nn.BCELoss()
    regression_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = scheduler_dict[scheduler_name](optimizer)
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    if isinstance(mix, tuple):
      mix_name = f'{mix[0]}_to_{mix[1]}'
    else:
      mix_name = mix
    writer = SummaryWriter(f'logs/{base_name}/{model.name()}_{mix_name}_{current_time}_lr={lr}_wd={wd}_{scheduler_name}_mixin={loss_mixin}_aug={data_augment}')
    # Train model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        classification_criterion=classification_criterion,
        regression_criterion=regression_criterion,
        loss_mixin=loss_mixin,
        optimizer=optimizer,
        device=device,
        writer=writer,
        epochs=epochs,
        scheduler=scheduler)
    perfs = evaluate_model(
        model=model,
        val_loader=val_loader,
        classification_criterion=classification_criterion,
        regression_criterion=regression_criterion,
        loss_mixin=loss_mixin,
        device=device
    )
    metrics.append({
        'pr_auc': perfs['val_pr_auc'],
        'roc_auc': perfs['val_roc_auc'],
        'pearson': perfs['val_pearson_r'],
        'mix': mix_name,
        'lr': lr,
        'wd': wd,
        'model': model.name()
    })

    writer.close()

  metrics = pd.DataFrame.from_records(metrics)

  if save_path:
    metrics.to_csv(save_path)

  return metrics


def evaluate_model(model,
                   val_loader,
                   classification_criterion,
                   regression_criterion,
                   loss_mixin,
                   device,
                   epoch = None,
                   writer = None):
  model.eval()
  class_probs = []
  class_preds = []
  regression_preds = []
  class_gt = []
  regression_gt = []
  total_loss = 0
  total_class_loss = 0
  total_regression_loss = 0

  with torch.no_grad():
    for sequences, class_labels, regression_labels in val_loader:
      sequences, class_labels, regression_labels = sequences.to(device), class_labels.to(device), regression_labels.to(device)

      class_outputs, regression_outputs = model(sequences)
      class_outputs, regression_outputs = class_outputs.squeeze(1), regression_outputs.squeeze(1)
      class_loss = classification_criterion(class_outputs, class_labels)
      regression_loss = regression_criterion(regression_outputs, regression_labels)
      loss = loss_mixin*class_loss + (1-loss_mixin)*regression_loss

      total_loss += loss.item()
      total_class_loss += class_loss.item()
      total_regression_loss += regression_loss.item()
      class_probs.extend(class_outputs.detach().cpu().numpy())
      class_preds.extend((class_outputs.detach() > 0.5).float().cpu().numpy())
      regression_preds.extend(regression_outputs.detach().cpu().numpy())
      class_gt.extend(class_labels.detach().cpu().numpy())
      regression_gt.extend(regression_labels.detach().cpu().numpy())

  accuracy = accuracy_score(class_gt, class_preds)
  pr_auc = average_precision_score(class_gt, class_probs)
  roc_auc = roc_auc_score(class_gt, class_probs)
  r, _ = pearsonr(regression_gt, regression_preds)

  if writer:
    writer.add_scalar('Loss/val', total_loss, epoch)
    writer.add_scalar('Class-Loss/val', total_class_loss, epoch)
    writer.add_scalar('Regression-Loss/val', total_regression_loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    writer.add_scalar('PR-AUC/val', pr_auc, epoch)
    writer.add_scalar('ROC-AUC/val', roc_auc, epoch)
    writer.add_scalar('Pearson/val', r, epoch)

  return {
    'val_loss': total_loss,
    'val_regression_loss': total_regression_loss,
    'val_class_loss': total_class_loss,
    'val_acc': accuracy,
    'val_pr_auc': pr_auc,
    'val_roc_auc': roc_auc,
    'val_pearson_r': r
  }


def train_model(model,
                train_loader,
                val_loader,
                classification_criterion,
                regression_criterion,
                loss_mixin,
                optimizer,
                device,
                writer,
                epochs=50,
                scheduler=None,
                patience: int = 5):
  
  dummy_input = next(iter(train_loader))[0][:1].to(device)
  _ = model(dummy_input)
  writer.add_graph(model, dummy_input)

  best_val_loss = float('inf')
  best_model_state = None

  for epoch in tqdm.tqdm(range(epochs)):
    # Training phase
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_regression_loss = 0
    class_probs = []
    class_preds = []
    regression_preds = []
    class_gt = []
    regression_gt = []

    for sequences, class_labels, regression_labels in train_loader:
      sequences, class_labels, regression_labels = sequences.to(device), class_labels.to(device), regression_labels.to(device)
      
      optimizer.zero_grad()
      class_outputs, regression_outputs = model(sequences)
      class_outputs, regression_outputs = class_outputs.squeeze(1), regression_outputs.squeeze(1)
      class_loss = classification_criterion(class_outputs, class_labels)
      regression_loss = regression_criterion(regression_outputs, regression_labels)
      loss = loss_mixin*class_loss + (1-loss_mixin)*regression_loss
      loss.backward()
      optimizer.step()
      
      total_loss += loss.item()
      total_class_loss += class_loss.item()
      total_regression_loss += regression_loss.item()
      class_probs.extend(class_outputs.detach().cpu().numpy())
      class_preds.extend((class_outputs.detach() > 0.5).float().cpu().numpy())
      regression_preds.extend(regression_outputs.detach().cpu().numpy())
      class_gt.extend(class_labels.detach().cpu().numpy())
      regression_gt.extend(regression_labels.detach().cpu().numpy())
    
    accuracy = accuracy_score(class_gt, class_preds)
    pr_auc = average_precision_score(class_gt, class_probs)
    roc_auc = roc_auc_score(class_gt, class_probs)
    r, _ = pearsonr(regression_gt, regression_preds)

    writer.add_scalar('Loss/train', total_loss, epoch)
    writer.add_scalar('Class-Loss/train', total_class_loss, epoch)
    writer.add_scalar('Regression-Loss/train', total_regression_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)
    writer.add_scalar('PR-AUC/train', pr_auc, epoch)
    writer.add_scalar('ROC-AUC/train', roc_auc, epoch)
    writer.add_scalar('Pearson/train', r, epoch)

    val_metrics = evaluate_model(
      model=model,
      val_loader=val_loader,
      classification_criterion=classification_criterion,
      regression_criterion=regression_criterion,
      loss_mixin=loss_mixin,
      device=device,
      epoch=epoch,
      writer=writer
    )

    val_loss = val_metrics['total_loss']

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_model_state = model.state_dict()
      epochs_no_improve = 0
    else:
      epochs_no_improve += 1

    if scheduler:
        scheduler.step()
    
    if epochs_no_improve >= patience:
      print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
      break

  if best_model_state:
    model.load_state_dict(best_model_state)

  return model