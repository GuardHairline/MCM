# modules/training_loop_fixed.py
"""
修复后的训练循环
主要修复：
1. TAM-CL损失重复计算问题
2. 梯度裁剪位置错误
3. 类别权重问题已通过label_config解决
4. 梯度处理顺序
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import time
import logging
from collections import Counter

from continual.metrics import ContinualMetrics, compute_metrics_example
from continual.label_embedding_manager import LabelEmbeddingManager
from continual.label_config import get_label_manager
from modules.evaluate import evaluate_single_task

logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, device, args, 
                ewc=None, fisher_selector=None, replay_memory=None,
                lwf=None, si=None, mas=None, gem=None,
                label_embedding_manager: Optional[LabelEmbeddingManager] = None,
                ddas_optimizer=None, logger_obj=None):
    """训练一个epoch（修复版）"""
    # 确保使用正确的任务头（使用非严格模式）
    if hasattr(model, 'set_active_head') and hasattr(args, 'session_name'):
        try:
            model.set_active_head(args.session_name, strict=False)
        except:
            pass  # 失败时使用当前head
    
    # 确保base_model的mode正确设置（重要！）
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'mode'):
        current_mode = getattr(args, 'mode', 'multimodal')
        model.base_model.mode = current_mode
        if logger_obj:
            logger_obj.info(f"Set base_model.mode to: {current_mode}")
    
    model.train()
    total_loss = 0.0
    total_samples = 0
    label_counter = Counter()
    ddas_feats = []
    
    # 获取任务信息
    task_config = get_label_manager().get_task_config(args.task_name)
    is_seq_task = task_config.task_type.value == "token" if task_config else False

    for batch_idx, batch in enumerate(train_loader):
        # 数据移到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        image_tensor = batch['image_tensor'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # ============ 前向传播 ============
        logits = None
        classification_loss = None
        fused_feat = None  # 确保fused_feat被初始化
        
        if args.tam_cl:
            # =========== TAM-CL 前向 =============
            out = model(input_ids, attention_mask, token_type_ids, image_tensor,
                        session_id=args.session_name)
            logits, seq, _ = out if isinstance(out, tuple) else (out, None, None)
            
            # 1. 分类损失
            class_weights = get_label_manager().get_class_weights(args.task_name, device)
            if is_seq_task:
                if class_weights is not None:
                    classification_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        weight=class_weights,
                        ignore_index=-100
                    )
                else:
                    classification_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=-100
                    )
            else:
                # 句级任务：应用类别权重
                class_weights = get_label_manager().get_class_weights(args.task_name, device)
                if class_weights is not None:
                    classification_loss = F.cross_entropy(logits, labels, weight=class_weights)
                else:
                    classification_loss = F.cross_entropy(logits, labels)
            
            # 2. 计算KD损失和多样性损失
            kd_loss = model.compute_distillation(seq, args.session_name, T=args.lwf_T) if seq is not None else 0.0
            div_loss = model.diversity_loss()
            
            # 3. 计算权重
            lambda_tam = args.old_sessions_count / (args.old_sessions_count + 1) if args.old_sessions_count > 0 else 0.0
            alpha_tam = getattr(args, "tam_alpha", args.lwf_alpha)
            
            # 4. 计算beta（需要detach避免梯度问题）
            if isinstance(kd_loss, torch.Tensor):
                beta_base = 0.1 * ((1 - lambda_tam) * classification_loss.detach() + lambda_tam * alpha_tam * kd_loss.detach())
            else:
                beta_base = 0.1 * (1 - lambda_tam) * classification_loss.detach()
            
            if isinstance(div_loss, torch.Tensor):
                beta_tam = torch.min(div_loss.detach(), beta_base)
            else:
                beta_tam = 0.0
            
            # 5. 最终损失（只计算一次！）
            loss = (1 - lambda_tam) * classification_loss
            if isinstance(kd_loss, torch.Tensor):
                loss = loss + lambda_tam * alpha_tam * kd_loss
            if isinstance(beta_tam, torch.Tensor) and isinstance(div_loss, torch.Tensor):
                loss = loss + beta_tam * div_loss
            
            label_counter.update(labels.cpu().numpy())
            
        elif args.clap4clip:
            # CLAP4CLIP 模型
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_tensor=image_tensor,
                task_name=args.session_name
            )
            # CLAP4CLIP: 根据任务类型应用类别权重
            is_seq_task = args.task_name in ["mate", "mner", "mabsa"]
            if is_seq_task:
                class_weights = get_label_manager().get_class_weights(args.task_name, device)
                if class_weights is not None:
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        weight=class_weights,
                        ignore_index=-100
                    )
                else:
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=-100
                    )
            else:
                class_weights = get_label_manager().get_class_weights(args.task_name, device)
                if class_weights is not None:
                    loss = F.cross_entropy(logits, labels, weight=class_weights)
                else:
                    loss = F.cross_entropy(logits, labels)
            label_counter.update(labels.cpu().numpy())
            
        else:
            # =========== 标准模型前向 =============
            # 检查是否使用混合头
            if hasattr(args, 'use_hierarchical_head') and args.use_hierarchical_head:
                token_logits, sent_logits = model(input_ids, attention_mask, token_type_ids, image_tensor)
                
                if is_seq_task:
                    # 序列标注任务：主要使用token头
                    class_weights = get_label_manager().get_class_weights(args.task_name, device)
                    if class_weights is not None:
                        main_loss = F.cross_entropy(
                            token_logits.reshape(-1, token_logits.size(-1)),
                            labels.reshape(-1),
                            weight=class_weights,
                            ignore_index=-100
                        )
                    else:
                        main_loss = F.cross_entropy(
                            token_logits.reshape(-1, token_logits.size(-1)),
                            labels.reshape(-1),
                            ignore_index=-100
                        )
                    
                    # 辅助损失：句子级预测
                    sentence_labels = torch.zeros(labels.size(0), dtype=torch.long, device=device)
                    for i in range(labels.size(0)):
                        seq_labels = labels[i]
                        valid_labels = seq_labels[seq_labels != -100]
                        if len(valid_labels) > 0 and (valid_labels != 0).any():
                            sentence_labels[i] = 1
                    
                    # 辅助损失：句子级别的二分类，不使用类别权重
                    aux_loss = F.cross_entropy(sent_logits, sentence_labels)
                    aux_weight = 0.1
                    loss = main_loss + aux_weight * aux_loss
                    logits = token_logits
                else:
                    # 句级分类任务：主要使用句子头，应用类别权重
                    class_weights = get_label_manager().get_class_weights(args.task_name, device)
                    if class_weights is not None:
                        main_loss = F.cross_entropy(sent_logits, labels, weight=class_weights)
                    else:
                        main_loss = F.cross_entropy(sent_logits, labels)
                    token_labels = labels.unsqueeze(1).expand(-1, token_logits.size(1))
                    aux_loss = F.cross_entropy(
                        token_logits.reshape(-1, token_logits.size(-1)),
                        token_labels.reshape(-1),
                        ignore_index=-100
                    )
                    aux_weight = 0.1
                    loss = main_loss + aux_weight * aux_loss
                    logits = sent_logits
                
                label_counter.update(labels.cpu().numpy())
                
                if args.ddas:
                    pooled_feature = token_logits.mean(dim=1)
            else:
                # 原有的单头逻辑
                if is_seq_task:
                    # 调试信息
                    if logger_obj and batch_idx == 0:
                        logger_obj.info(f"DEBUG: Training sequence task, base_model.mode={getattr(model.base_model, 'mode', 'unknown')}")
                    
                    fused_feat = model.base_model(
                        input_ids, attention_mask, token_type_ids, image_tensor,
                        return_sequence=True
                    )
                    
                    # 检查fused_feat是否为None
                    if fused_feat is None:
                        error_msg = f"ERROR: fused_feat is None! base_model.mode={getattr(model.base_model, 'mode', 'unknown')}, task={args.task_name}"
                        if logger_obj:
                            logger_obj.error(error_msg)
                        raise ValueError(error_msg)
                    
                    logits = model.head(fused_feat)
                    
                    class_weights = get_label_manager().get_class_weights(args.task_name, device)
                    if class_weights is not None:
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            labels.reshape(-1),
                            weight=class_weights,
                            ignore_index=-100
                        )
                    else:
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            labels.reshape(-1),
                            ignore_index=-100
                        )
                    
                    if args.ddas:
                        pooled_feature = fused_feat.mean(dim=1)
                else:
                    fused_feat = model.base_model(
                        input_ids, attention_mask, token_type_ids, image_tensor,
                        return_sequence=False
                    )
                    logits = model.head(fused_feat)
                    # 句级任务：应用类别权重
                    class_weights = get_label_manager().get_class_weights(args.task_name, device)
                    if class_weights is not None:
                        loss = F.cross_entropy(logits, labels, weight=class_weights)
                    else:
                        loss = F.cross_entropy(logits, labels)
                    
                    if args.ddas:
                        pooled_feature = fused_feat
                    
                    label_counter.update(labels.cpu().numpy())
            
            if args.ddas:
                ddas_feats.append(pooled_feature.detach())
        
        # ============ 添加正则化损失 ============
        # 标签相似度正则化
        if label_embedding_manager:
            similarity_loss = label_embedding_manager.get_similarity_loss()
            if isinstance(similarity_loss, torch.Tensor) and similarity_loss.requires_grad:
                loss = loss + similarity_loss
        
        # EWC 正则化
        if ewc is not None:
            ewc_loss = ewc.penalty(model)
            if isinstance(ewc_loss, torch.Tensor) and ewc_loss.requires_grad:
                loss = loss + ewc_loss
        
        # MyMethod 正则化
        if fisher_selector is not None:
            mymethod_loss = fisher_selector.penalty(model)
            if isinstance(mymethod_loss, torch.Tensor) and mymethod_loss.requires_grad:
                loss = loss + mymethod_loss
        
        # LwF 蒸馏损失
        if lwf is not None and logits is not None:
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "image_tensor": image_tensor
            }
            lwf_loss = lwf.distillation_loss(logits, inputs)
            if isinstance(lwf_loss, torch.Tensor) and lwf_loss.requires_grad:
                loss = loss + lwf_loss
        
        # SI 正则化
        if si is not None:
            si_loss = si.penalty()
            if isinstance(si_loss, torch.Tensor) and si_loss.requires_grad:
                loss = loss + si_loss
        
        # MAS 正则化
        if mas is not None:
            mas_loss = mas.penalty()
            if isinstance(mas_loss, torch.Tensor) and mas_loss.requires_grad:
                loss = loss + mas_loss
        
        # Experience Replay（修复版：在batch训练时集成）
        if replay_memory and hasattr(args, 'replay') and args.replay:
            # 每隔几个batch进行一次replay
            if batch_idx % getattr(args, 'replay_frequency', 4) == 0:
                replay_session = replay_memory.sample_replay_session(batch_idx, model, device, args)
                if replay_session is not None:
                    replay_loss = replay_memory.compute_replay_loss(replay_session, model, device)
                    if replay_loss is not None:
                        replay_weight = getattr(args, 'replay_weight', 0.5)
                        loss = loss + replay_weight * replay_loss
                        if logger_obj and batch_idx % 10 == 0:
                            logger_obj.debug(f"Added replay loss from '{replay_session}': {replay_loss.item():.4f}")
        
        # ✓ MoE Load Balancing Loss（修复版）
        if args.moe_adapters and hasattr(model, 'base_model') and hasattr(model.base_model, 'text_adapters'):
            balance_coef = getattr(args, 'moe_balance_coef', 0.01)  # 可配置的系数
            bal_loss = 0.0
            num_layers = 0
            
            # 收集所有MoE层的load_loss
            for moe_layer in model.base_model.text_adapters + model.base_model.image_adapters:
                if hasattr(moe_layer, 'load_loss') and moe_layer.load_loss is not None:
                    bal_loss += moe_layer.load_loss
                    num_layers += 1
            
            # 平均并加权
            if num_layers > 0:
                bal_loss = bal_loss / num_layers  # 平均
                loss = loss + balance_coef * bal_loss
                
                if logger_obj and batch_idx % 50 == 0:
                    logger_obj.debug(f"MoE Balance Loss: {bal_loss.item():.4f} (coef={balance_coef})")
        
        # ============ 反向传播和梯度处理 ============
        # 检查loss是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Batch {batch_idx}: Invalid loss detected (NaN or Inf), skipping batch")
            continue
        
        loss.backward()
        
        # 梯度裁剪（在GEM投影之前）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # GEM 梯度投影（在梯度裁剪之后，优化器更新之前）
        if gem is not None:
            gem.project_gradients()
        
        optimizer.step()
        
        # SI 梯度累积（在optimizer.step()之后）
        if si is not None:
            si.accumulate()
        
        # === DDAS 自编码器训练 ===
        if args.moe_adapters and args.ddas and ddas_optimizer is not None and ddas_feats:
            ae_inputs = torch.cat(ddas_feats, dim=0)
            ddas_optimizer.zero_grad()
            recon = model.ddas.ae_list[-1](ae_inputs)
            ae_loss = F.mse_loss(recon, ae_inputs)
            ae_loss.backward()
            ddas_optimizer.step()
            ddas_feats.clear()
        
        total_loss += loss.item()
        total_samples += input_ids.size(0)
        
        if logger_obj and batch_idx % 10 == 0:
            logger_obj.info(f"Batch {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    if logger_obj:
        logger_obj.info(f"Epoch total_loss={total_loss:.4f}, avg_loss={avg_loss:.4f}, total_samples={total_samples}")
    
    return avg_loss


def validate_epoch(model, val_loader, device, args, logger=None):
    """验证一个epoch"""
    # 确保使用正确的任务头（使用非严格模式）
    if hasattr(model, 'set_active_head') and hasattr(args, 'session_name'):
        try:
            model.set_active_head(args.session_name, strict=False)
        except:
            pass  # 失败时使用当前head
    
    # 确保base_model的mode正确设置
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'mode'):
        current_mode = getattr(args, 'mode', 'multimodal')
        model.base_model.mode = current_mode
    
    # CLAP4CLIP模型需要特殊处理
    if args.clap4clip and hasattr(model, 'set_current_task'):
        model.set_current_task(args.session_name)
    
    # 直接用 evaluate_single_task 评估
    metrics = evaluate_single_task(model, args.task_name, "dev", device, args)
    return None, metrics  # 返回None作为loss（因为我们不计算验证loss）


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, args,
                ewc=None, fisher_selector=None, replay_memory=None,
                lwf=None, si=None, mas=None, gem=None,
                label_embedding_manager: Optional[LabelEmbeddingManager] = None,
                logger=None):
    """完整训练流程（修复版）"""
    best_val_metric = 0.0  # 改用accuracy作为标准
    best_metrics = None
    patience = args.patience
    no_improve_count = 0
    
    # 收集每个epoch的loss和dev metrics
    epoch_losses = []
    dev_metrics_history = []
    
    # 创建DDAS优化器
    ddas_optimizer = None
    if args.ddas and hasattr(model, 'ddas') and model.ddas is not None:
        ddas_optimizer = torch.optim.Adam(model.ddas.parameters(), lr=1e-4)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        if logger:
            logger.info(f"{'='*80}")
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
            logger.info(f"{'='*80}")
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, device, args,
            ewc, fisher_selector, replay_memory, lwf, si, mas, gem,
            label_embedding_manager, ddas_optimizer, logger
        )
                
        # 验证
        val_loss, metrics = validate_epoch(model, val_loader, device, args, logger)
        
        # 记录loss和metrics
        epoch_losses.append(train_loss)
        dev_metrics_history.append(metrics)
        
        # Early stopping检查
        current_metric = metrics.get('acc', 0.0)
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_metrics = metrics.copy()
            no_improve_count = 0
            if logger:
                logger.info(f"✓ New best accuracy: {best_val_metric:.4f}")
        else:
            no_improve_count += 1
            if logger:
                logger.info(f"✗ No improvement for {no_improve_count} epoch(s)")
        
        # 学习率调度
        if scheduler:
            scheduler.step()
        
        epoch_time = time.time() - start_time
        
        if logger:
            logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Acc: {current_metric:.4f}, "
                       f"Time: {epoch_time:.2f}s")
        
        # Early stopping
        if no_improve_count >= patience:
            if logger:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # 训练结束后评估dev/test
    final_dev_metrics = evaluate_single_task(model, args.task_name, "dev", device, args)
    final_test_metrics = evaluate_single_task(model, args.task_name, "test", device, args)
    
    if args.ddas and hasattr(model, 'ddas') and model.ddas is not None:
        model.ddas.add_task()
    
    if best_metrics is None:
        best_metrics = final_dev_metrics
    
    return {
        "best_metrics": best_metrics,
        "epoch_losses": epoch_losses,
        "dev_metrics_history": dev_metrics_history,
        "final_dev_metrics": final_dev_metrics,
        "final_test_metrics": final_test_metrics
    }


def update_continual_learning_components(model, train_loader, device, args, 
                                       ewc=None, fisher_selector=None, si=None, mas=None, gem=None,
                                       session_info=None, logger=None):
    """更新持续学习组件"""
    # EWC Fisher估计（使用更大的样本量）
    if ewc:
        ewc.estimate_and_save_fisher(train_loader, device=device, sample_size=500)
        if logger:
            logger.info(f"[EWC] Fisher estimated and saved (500 samples)")
    
    # MyMethod Fisher估计（使用更大的样本量）
    if fisher_selector:
        fisher_selector.estimate_and_save_fisher(train_loader, device=device, sample_size=500)
        if logger:
            logger.info(f"[MyMethod] Fisher estimated and saved (500 samples)")
    
    # SI Omega更新
    if si:
        si.update_omega()
        if logger:
            logger.info(f"[SI] Omega updated")
    
    # MAS Importance计算
    if mas:
        mas.compute_importance(train_loader, device, task_name=args.task_name)
        if logger:
            logger.info(f"[MAS] Importance computed for task '{args.task_name}'")
    
    # GEM Memory保存
    if gem:
        gem.save_memory(args.task_name)
        if logger:
            logger.info(f"[GEM] Memory for task '{args.task_name}' saved")
    
    return session_info

