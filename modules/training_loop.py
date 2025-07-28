# modules/training_loop.py
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
from .train_utils import get_class_weights, is_sequence_task
from modules.evaluate import evaluate_single_task


def train_epoch(model, train_loader, optimizer, device, args, 
                ewc=None, fisher_selector=None, replay_memory=None,
                lwf=None, si=None, mas=None, gem=None,
                label_embedding_manager: Optional[LabelEmbeddingManager] = None,
                ddas_optimizer=None, logger=None):
    """训练一个epoch"""
    # 确保使用正确的任务头
    if hasattr(model, 'set_active_head') and hasattr(args, 'session_name'):
        model.set_active_head(args.session_name)
    
    model.train()
    total_loss = 0.0
    total_samples = 0
    label_counter = Counter()
    ddas_feats = []

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
        
        # 前向传播
        if args.tam_cl:
            # =========== TAM-CL 前向 =============
            out = model(input_ids, attention_mask, token_type_ids, image_tensor,
                        session_id=args.session_name)
            logits, seq, _ = out if isinstance(out, tuple) else (out, None, None)
            # 1. 分类损失
            classification_loss = ...
            # 2. KD + 多样性损失
            kd_loss = model.compute_distillation(seq, args.session_name, T=args.lwf_T)
            div_loss = model.diversity_loss()
            # 3. 权重 λ、α、β 计算
            lambda_tam = (len(train_info["sessions"])) / (len(train_info["sessions"]) + 1)
            alpha_tam = getattr(args, "tam_alpha", args.lwf_alpha)
            beta_base = 0.1 * ((1 - lambda_tam) * classification_loss + lambda_tam * alpha_tam * kd_loss)
            beta_tam = torch.min(div_loss.detach(), beta_base.detach())
            # 4. 组合
            loss = (1 - lambda_tam) * classification_loss + lambda_tam * alpha_tam * kd_loss + beta_tam * div_loss
        elif args.clap4clip:
            # CLAP4CLIP 模型直接处理
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_tensor=image_tensor,
                task_name=args.session_name
            )
            loss = F.cross_entropy(logits, labels)
            label_counter.update(labels.cpu().numpy())
        else:
            # 检查是否使用混合头
            if hasattr(args, 'use_hierarchical_head') and args.use_hierarchical_head:
                # 使用混合头模型
                token_logits, sent_logits = model(input_ids, attention_mask, token_type_ids, image_tensor)
                
                is_seq_task = is_sequence_task(args.task_name)
                
                if is_seq_task:
                    # 序列标注任务：主要使用token头，句子头作为辅助监督
                    # 主要损失：token级预测
                    class_weights = get_class_weights(args.task_name, device)
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
                    
                    # 辅助损失：句子级预测（将token标签聚合为句子标签）
                    # 简单的聚合策略：如果序列中有任何非O标签，则句子标签为1，否则为0
                    sentence_labels = torch.zeros(labels.size(0), dtype=torch.long, device=device)
                    for i in range(labels.size(0)):
                        seq_labels = labels[i]
                        # 移除padding标签(-100)
                        valid_labels = seq_labels[seq_labels != -100]
                        if len(valid_labels) > 0 and (valid_labels != 0).any():
                            sentence_labels[i] = 1
                    
                    aux_loss = F.cross_entropy(sent_logits, sentence_labels)
                    
                    # 总损失 = 主要损失 + 辅助损失权重
                    aux_weight = 0.1  # 辅助损失权重
                    loss = main_loss + aux_weight * aux_loss
                    
                    logits = token_logits  # 用于后续处理
                else:
                    # 句级分类任务：主要使用句子头，token头作为辅助监督
                    # 主要损失：句子级预测
                    main_loss = F.cross_entropy(sent_logits, labels)
                    
                    # 辅助损失：token级预测（将句子标签扩展到所有token）
                    token_labels = labels.unsqueeze(1).expand(-1, token_logits.size(1))
                    aux_loss = F.cross_entropy(
                        token_logits.reshape(-1, token_logits.size(-1)),
                        token_labels.reshape(-1),
                        ignore_index=-100
                    )
                    
                    # 总损失 = 主要损失 + 辅助损失权重
                    aux_weight = 0.1  # 辅助损失权重
                    loss = main_loss + aux_weight * aux_loss
                    
                    logits = sent_logits  # 用于后续处理
                
                label_counter.update(labels.cpu().numpy())
                
                if args.ddas:
                    # 对于混合头，使用token特征的均值作为DDAS特征
                    pooled_feature = token_logits.mean(dim=1)
            else:
                # 原有的单头逻辑
                is_seq_task = is_sequence_task(args.task_name)
                
                if is_seq_task:
                    # 序列标注任务
                    fused_feat = model.base_model(
                        input_ids, attention_mask, token_type_ids, image_tensor,
                        return_sequence=True
                    )
                    logits = model.head(fused_feat)

                    # 类别权重处理
                    class_weights = get_class_weights(args.task_name, device)
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
                    # 句级分类任务
                    fused_feat = model.base_model(
                        input_ids, attention_mask, token_type_ids, image_tensor,
                        return_sequence=False
                    )
                    logits = model.head(fused_feat)
                    loss = F.cross_entropy(logits, labels)
                    
                    if args.ddas:
                        pooled_feature = fused_feat
                    
                    label_counter.update(labels.cpu().numpy())
            
            if args.ddas:
                ddas_feats.append(pooled_feature.detach())
        
        # 添加标签相似度正则化损失
        if label_embedding_manager:
            similarity_loss = label_embedding_manager.get_similarity_loss()
            loss += similarity_loss
        
        # EWC 正则化
        if ewc is not None:
            ewc_loss = ewc.penalty(model)
            loss += ewc_loss
        
        # MyMethod 正则化
        if fisher_selector is not None:
            mymethod_loss = fisher_selector.penalty(model)
            loss += mymethod_loss
        
        # LwF 蒸馏损失
        if lwf is not None:
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "image_tensor": image_tensor
            }
            lwf_loss = lwf.distillation_loss(logits, inputs)
            loss += lwf_loss
        
        # SI 正则化
        if si is not None:
            si_loss = si.penalty()
            loss += si_loss
        
        # MAS 正则化
        if mas is not None:
            mas_loss = mas.penalty()
            loss += mas_loss
        
        # TAM-CL 特殊处理
        if args.tam_cl:
            # 保存 inputs 供模型内部 distillation 调用
            model.last_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "image_tensor": image_tensor
            }
            out = model(input_ids, attention_mask, token_type_ids, image_tensor, args.session_name)
            if isinstance(out, tuple) and len(out) == 3:
                _, seq, _ = out
            else:
                seq = None
            if seq is not None:
                kd_loss = model.compute_distillation(seq, args.session_name, T=args.lwf_T)
                div_loss = model.diversity_loss()
                loss = loss + args.lwf_alpha * kd_loss + 0.1 * div_loss
        
        # MoE 路由平衡损失
        if args.moe_adapters and hasattr(model, 'base_model') and hasattr(model.base_model, 'text_adapters'):
            balance_coef = 0.01
            bal_loss = 0.0
            for moe_layer in model.base_model.text_adapters + model.base_model.image_adapters:
                with torch.no_grad():
                    pooled = fused_feat.mean(dim=1) if fused_feat.dim() == 3 else fused_feat
                probs = moe_layer.softmax(moe_layer.router(pooled))
                bal_loss += (probs.mean(dim=0) ** 2).sum()
            loss = loss + balance_coef * bal_loss
        
        # 反向传播
        loss.backward()

        # # [DEBUG] 打印logits和labels的shape及前几个数值，排查loss计算问题
        # if 'logits' in locals() and 'labels' in locals():
        #     try:
        #         print(f"[DEBUG] logits.shape: {logits.shape}, labels.shape: {labels.shape}")
        #         print(f"[DEBUG] logits sample: {logits.flatten()[:10].detach().cpu().numpy()}")
        #         print(f"[DEBUG] labels sample: {labels.flatten()[:10].detach().cpu().numpy()}")
        #     except Exception as e:
        #         print(f"[DEBUG] logits/labels print error: {e}")
        
        # # [DEBUG] 打印base_model、head、label embedding的梯度norm
        # if hasattr(model, "base_model"):
        #     for n, p in model.base_model.named_parameters():
        #         if p.grad is not None:
        #             print(f"[DEBUG] base_model.{n} grad norm: {p.grad.norm().item():.6f}")
        #         break  # 只打印第一个参数
        # if hasattr(model, "head"):
        #     for n, p in model.head.named_parameters():
        #         if p.grad is not None:
        #             print(f"[DEBUG] head.{n} grad norm: {p.grad.norm().item():.6f}")
        #         break
        # # label embedding grad
        # emb = None
        # if hasattr(model, "head") and hasattr(model.head, "label_emb"):
        #     emb = model.head.label_emb
        # if emb is not None and hasattr(emb, "embedding") and hasattr(emb.embedding, "weight"):
        #     grad = emb.embedding.weight.grad
        #     if grad is not None:
        #         print(f"[DEBUG] label embedding grad norm: {grad.norm().item():.6f}")
        #         # 如果有freeze，打印新任务label embedding的梯度
        #         if hasattr(emb, "label2idx") and hasattr(emb, "task_name"):
        #             task_key = emb.task_name if hasattr(emb, "task_name") else None
        #             num_labels = getattr(emb, "num_labels", None)
        #             if task_key and num_labels:
        #                 try:
        #                     indices = [emb.label2idx[(task_key, i)] for i in range(num_labels)]
        #                     print(f"[DEBUG] new task label embedding grad: {grad[indices]}")
        #                 except Exception as e:
        #                     print(f"[DEBUG] label embedding grad index error: {e}")
        #     else:
        #         print("[DEBUG] label embedding grad is None")
        
        # GEM 梯度投影
        if gem is not None:
            current_grads = [p.grad for _, p in model.named_parameters() if p.grad is not None]
            gem.project_gradients(None, current_grads)
        
        # SI 梯度累积
        if si is not None:
            grads = {n: p.grad.clone().detach() for n, p in model.named_parameters() if p.grad is not None}
            si.accumulate(grads)
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # === DDAS 自编码器训练 ===
        if args.ddas and ddas_optimizer is not None and ddas_feats:
            # 将当前 batch 所有特征拼接
            ae_inputs = torch.cat(ddas_feats, dim=0)        # (N, D)
            ddas_optimizer.zero_grad()
            recon = model.ddas.ae_list[-1](ae_inputs)       # 只训练最后一个自编码器
            ae_loss = F.mse_loss(recon, ae_inputs)
            ae_loss.backward()
            ddas_optimizer.step()
            # 清空特征缓存
            ddas_feats.clear()
        
        total_loss += loss.item()
        total_samples += input_ids.size(0)
        print(f"Batch {batch_idx}: loss={loss.item()}")
    print(f"Epoch total_loss={total_loss}, total_samples={total_samples}")
    return total_loss / total_samples


def validate_epoch(model, val_loader, device, args, logger=None):
    """验证一个epoch"""
    # 确保使用正确的任务头
    if hasattr(model, 'set_active_head') and hasattr(args, 'session_name'):
        model.set_active_head(args.session_name)
    
    # CLAP4CLIP模型需要特殊处理
    if args.clap4clip and hasattr(model, 'set_current_task'):
        model.set_current_task(args.session_name)
    
    # 直接用 evaluate_single_task 评估
    metrics = evaluate_single_task(model, args.task_name, "dev", device, args)
    avg_loss = None  # 如果需要平均loss，可在 evaluate_single_task 里返回或单独实现
    # if logger:
    #     logger.info(f"Validation Metrics: {metrics}")
    return avg_loss, metrics


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, args,
                ewc=None, fisher_selector=None, replay_memory=None,
                lwf=None, si=None, mas=None, gem=None,
                label_embedding_manager: Optional[LabelEmbeddingManager] = None,
                logger=None):
    """完整训练流程"""
    best_val_loss = float('inf')
    best_metrics = None
    patience = args.patience
    no_improve_count = 0
    
    # 新增：收集每个epoch的loss和dev metrics
    epoch_losses = []
    dev_metrics_history = []
    
    # 创建DDAS优化器
    ddas_optimizer = None
    if args.ddas and hasattr(model, 'ddas') and model.ddas is not None:
        ddas_optimizer = torch.optim.Adam(model.ddas.parameters(), lr=1e-4)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, device, args,
            ewc, fisher_selector, replay_memory, lwf, si, mas, gem,
            label_embedding_manager, ddas_optimizer, logger
        )
        
        # 经验重放
        if args.replay and replay_memory and replay_memory.do_replay(epoch + 1, model, device, args):
            replay_session_name = replay_memory.sample_replay_session(epoch + 1, model, device, args)
            if replay_session_name is not None:
                replay_loss = replay_memory.run_replay_step(replay_session_name, model, epoch + 1, device, args)
                if logger:
                    logger.info(f"Replay loss: {replay_loss.item():.4f}")
        
        # 验证
        val_loss, metrics = validate_epoch(model, val_loader, device, args, logger)
        
        # 新增：记录loss和metrics
        epoch_losses.append(train_loss)
        dev_metrics_history.append(metrics)
        
        # 学习率调度
        if scheduler:
            scheduler.step()
        
        epoch_time = time.time() - start_time
        
        if val_loss is not None:
            logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Time: {epoch_time:.2f}s")
        else:
            logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Time: {epoch_time:.2f}s "
                        f"Metrics: {metrics}")
    # 新增：训练结束后评估一次dev/test
    final_dev_metrics = evaluate_single_task(model, args.task_name, "dev", device, args)
    final_test_metrics = evaluate_single_task(model, args.task_name, "test", device, args)
    if args.ddas and hasattr(model, 'ddas') and model.ddas is not None:
        # 冻结旧的自编码器并为新任务添加一个自编码器
        model.ddas.add_task()
    return {
        "best_metrics": best_metrics,
        "epoch_losses": epoch_losses,
        "dev_metrics_history": dev_metrics_history,
        "final_dev_metrics": final_dev_metrics,
        "final_test_metrics": final_test_metrics
    }


def evaluate_all_tasks(model, test_loaders, device, args, logger=None):
    """评估所有已学习的任务"""
    model.eval()
    all_metrics = {}
    
    with torch.no_grad():
        for task_name, test_loader in test_loaders.items():
            if logger:
                logger.info(f"Evaluating task: {task_name}")
            
            all_predictions = []
            all_labels = []
            
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids', None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                image_tensor = batch['image_tensor'].to(device)
                labels = batch['labels'].to(device)
                
                # 检查是否为特殊模型类型
                if args.tam_cl:
                    out = model(
                        input_ids, attention_mask, token_type_ids, image_tensor,
                        session_id=task_name
                    )
                    if isinstance(out, tuple) and len(out) == 3:
                        logits, _, _ = out
                    else:
                        logits = out
                elif args.clap4clip:
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        image_tensor=image_tensor,
                        task_name=task_name
                    )
                else:
                    is_seq_task = is_sequence_task(task_name)
                
                    if is_seq_task:
                        fused_feat = model.base_model(
                            input_ids, attention_mask, token_type_ids, image_tensor,
                            return_sequence=True
                        )
                        logits = model.head(fused_feat)
                    else:
                        fused_feat = model.base_model(
                            input_ids, attention_mask, token_type_ids, image_tensor,
                            return_sequence=False
                        )
                        logits = model.head(fused_feat)
                
                if is_sequence_task(task_name):
                    predictions = torch.argmax(logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
                else:
                    predictions = torch.argmax(logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            metrics = compute_metrics_example(all_predictions, all_labels, task_name)
            all_metrics[task_name] = metrics
            
            if logger:
                logger.info(f"Task {task_name} metrics: {metrics}")
    
    return all_metrics 


def update_continual_learning_components(model, train_loader, device, args, 
                                       ewc=None, fisher_selector=None, si=None, gem=None,
                                       session_info=None, logger=None):
    """更新持续学习组件"""
    # EWC Fisher估计
    if ewc:
        ewc.estimate_and_save_fisher(train_loader, device=device, sample_size=200)
        if logger:
            logger.info(f"[EWC] Fisher estimated and saved")
    
    # MyMethod Fisher估计
    if fisher_selector:
        fisher_selector.estimate_and_save_fisher(train_loader, device=device, sample_size=200)
        if logger:
            logger.info(f"[MyMethod] Fisher estimated and saved")
    
    # SI Omega更新
    if si:
        si.update_omega()
        if logger:
            logger.info(f"[SI] Omega updated")
    
    # GEM Memory保存
    if gem:
        gem.save_memory(args.task_name)
        if logger:
            logger.info(f"[GEM] Memory for task '{args.task_name}' saved")
    
    return session_info 