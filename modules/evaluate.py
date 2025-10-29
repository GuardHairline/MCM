# modules/evaluate.py
from collections import Counter


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from datasets.get_dataset import get_dataset
from utils.decode import decode_mate, decode_mner, decode_mabsa
from continual.moe_adapters.ddas_router import DDASRouter
import logging

logger = logging.getLogger("evaluate")


def _need_ddas(args):
    """统一判断当前会话是否启用 DDAS。"""
    if isinstance(args, dict):
        return bool(args.get("ddas", 0))
    return bool(getattr(args, "ddas", 0))
def evaluate_single_task(model, task_name, split, device, args):
    """
    对指定任务的 {split} (dev/test) 数据集进行评估，返回准确率(%)。
    """
    # 确保使用正确的任务头（使用非严格模式）
    if hasattr(model, 'set_active_head') and hasattr(args, 'session_name'):
        try:
            # 使用strict=False允许失败时继续
            model.set_active_head(args.session_name, strict=False)
        except Exception as e:
            # 如果设置活动头失败（比如在0样本检测时），使用默认行为
            logger.warning(f"Failed to set active head for session {args.session_name}: {e}")
            # 不设置活动头，让模型使用默认行为
    
    # 为CLAP4CLIP模型设置当前任务
    if hasattr(model, 'set_current_task') and hasattr(args, 'session_name'):
        try:
            model.set_current_task(args.session_name)
        except Exception as e:
            logger.warning(f"Failed to set current task for session {args.session_name}: {e}")
    
    # 读取通用参数
    if isinstance(args, dict):
        batch_size = args.get("batch_size")
        mode = args.get("mode")
    else:
        batch_size = args.batch_size
        mode = args.mode

    model.base_model.mode = mode
    use_ddas = _need_ddas(args) and getattr(model, "ddas", None) is not None

    # 数据
    ds = get_dataset(task_name, split, args)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()

    is_sequence_task = (task_name in ["mate", "mner", "mabsa"])

    # 指标积累勇气
    # 用于 token-level (或句级) 计算
    all_preds_token = []
    all_labels_token = []

    # 用于 chunk-level 计算
    all_chunks_pred = []
    all_chunks_gold = []

    #
    # # ======== 新增：用于只打印前 n 条 Debug ========
    # debug_print_limit = 20
    # debug_print_count = 0
    #
    # # 获取 tokenizer 如果需要可视化 token
    # # (这里假设 model.base_model.text_encoder 是一个 huggingface AutoModel,
    # #  需要你自己根据项目结构拿到 tokenizer 或在 dataset 里存.
    # #  如果此处无法直接拿到, 也可在 dataset 那边保留 'raw_text' 用于打印)
    # tokenizer = None
    # if hasattr(model.base_model, "text_encoder") and hasattr(model.base_model.text_encoder, "config"):
    #     from transformers import AutoTokenizer
    #     tokenizer_name = model.base_model.text_encoder.name_or_path  # 可能要看看属性是什么
    #     try:
    #         tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    #     except:
    #         tokenizer = None

    label_counter = Counter()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            image_tensor = batch["image_tensor"].to(device)
            labels = batch["labels"].to(device)


            if is_sequence_task:
                # 统计标签数量（排除-100）
                for label in labels.view(-1).cpu().tolist():
                    if label != -100:
                        label_counter[label] += 1

                # 检查是否为特殊模型类型
                if hasattr(args, 'tam_cl') and args.tam_cl:
                    # TAM-CL 模型
                    try:
                        out = model(input_ids, attention_mask, token_type_ids, image_tensor, session_id=args.session_name)
                        if isinstance(out, tuple) and len(out) == 3:
                            logits, _, _ = out
                        else:
                            logits = out
                    except Exception as e:
                        logger.warning(f"TAM-CL forward failed, using fallback: {e}")
                        # 使用fallback方法
                        fused_feat = model.base_model(input_ids, attention_mask, token_type_ids, image_tensor,
                                                      return_sequence=True)
                        logits = model.head(fused_feat)
                elif hasattr(args, 'clap4clip') and args.clap4clip:
                    # CLAP4CLIP 模型
                    try:
                        logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            image_tensor=image_tensor,
                            task_name=args.session_name
                        )
                    except Exception as e:
                        logger.warning(f"CLAP4CLIP forward failed, using fallback: {e}")
                        # 使用fallback方法
                        fused_feat = model.base_model(input_ids, attention_mask, token_type_ids, image_tensor,
                                                      return_sequence=True)
                        logits = model.head(fused_feat)
                else:
                    # 标准模型
                    try:
                        fused_feat = model.base_model(input_ids, attention_mask, token_type_ids, image_tensor,
                                                      return_sequence=True)
                        if use_ddas:
                            # 用 CLS 平均表示决定是否替换
                            pooled = fused_feat.mean(dim=1)           # (B,H)
                            branch_mask, _ = model.ddas(pooled)
                            if (~branch_mask).any():
                                # 重新用冻结主干特征
                                plain_feat = model.base_model.base_model(
                                    input_ids[~branch_mask], attention_mask[~branch_mask],
                                    token_type_ids[~branch_mask], image_tensor[~branch_mask],
                                    return_sequence=True
                                )
                                fused_feat[~branch_mask] = plain_feat
                        
                        # 检查head类型，确保序列任务使用正确的head
                        if hasattr(model, 'head'):
                            try:
                                logits = model.head(fused_feat)              # (B,L,C)
                            except ValueError as e:
                                if "not enough values to unpack" in str(e):
                                    # 如果head期望2维输入但得到3维，说明head类型不匹配
                                    logger.warning(f"Head type mismatch for task {task_name}, using fallback")
                                    # 创建一个简单的线性分类器作为fallback
                                    if not hasattr(model, '_fallback_head'):
                                        model._fallback_head = nn.Linear(fused_feat.size(-1), args.num_labels).to(fused_feat.device)
                                    # 对于序列任务，取平均池化
                                    pooled_feat = fused_feat.mean(dim=1)  # (B, H)
                                    logits = model._fallback_head(pooled_feat)
                                else:
                                    raise e
                        else:
                            raise ValueError("Model has no head")
                    except Exception as e:
                        logger.warning(f"Standard model forward failed: {e}")
                        # 如果head失败，尝试使用默认head
                        if hasattr(model, 'head'):
                            logits = model.head(fused_feat)
                        else:
                            raise e
                
                # logits: [batch_size, seq_len, num_labels]
                # preds => [batch_size, seq_len]
                preds = torch.argmax(logits, dim=2)

                # 将它们 flatten (含-100) 以计算 token-level 参考指标
                all_preds_token.extend(preds.view(-1).cpu().tolist())
                all_labels_token.extend(labels.view(-1).cpu().tolist())

                # 开始做 chunk-level decode
                # print(f"preds.shape: {preds.shape}")
                if preds.dim() == 2:
                    bsz, seqlen = preds.shape
                elif preds.dim() == 3:
                    bsz, seqlen, num_labels = preds.shape
                elif preds.dim() == 4:
                    bsz, seqlen, _, num_labels = preds.shape  # 例如，处理跨度任务的情况
                else:
                    raise ValueError(f"Unexpected preds dimension: {preds.dim()}")

                for i in range(bsz):
                    # 过滤 -100
                    valid_len = (labels[i] != -100).sum().item()
                    pred_i = preds[i, :valid_len].cpu().tolist()
                    gold_i = labels[i, :valid_len].cpu().tolist()

                    if task_name == "mate":
                        pred_chunks = decode_mate(pred_i)
                        gold_chunks = decode_mate(gold_i)
                    elif task_name == "mner":
                        pred_chunks = decode_mner(pred_i)
                        gold_chunks = decode_mner(gold_i)
                    elif task_name == "mabsa":
                        pred_chunks = decode_mabsa(pred_i)
                        gold_chunks = decode_mabsa(gold_i)
                    else:
                        pred_chunks = set()
                        gold_chunks = set()

                    all_chunks_pred.append(pred_chunks)
                    all_chunks_gold.append(gold_chunks)

                    # # ========== Debug Print: 只打印前 debug_print_limit 条 ==========
                    # if debug_print_count < debug_print_limit:
                    #     debug_print_count += 1
                    #     logger.info(f"\n========== [DEBUG] Sample #{debug_print_count}  ==========")
                    #
                    #     # 如果能拿到 tokenizer，就把 input_ids -> tokens
                    #     if tokenizer:
                    #         # 取对应的 input_ids[i,:valid_len]
                    #         input_ids_i = input_ids[i, :valid_len].cpu().tolist()
                    #         tokens = tokenizer.convert_ids_to_tokens(input_ids_i, skip_special_tokens=False)
                    #         # 打印 tokens + gold + pred
                    #         logger.info("Tokens:")
                    #         for tk_idx, tk in enumerate(tokens):
                    #             logger.info(f"  idx={tk_idx}, token={tk}, gold={gold_i[tk_idx]}, pred={pred_i[tk_idx]}")
                    #     else:
                    #         logger.info(f"input_ids: {input_ids[i, :valid_len].cpu().tolist()}")
                    #         logger.info(f"gold: {gold_i}")
                    #         logger.info(f"pred: {pred_i}")
                    #
                    #     logger.info(f"Gold chunks: {gold_chunks}")
                    #     logger.info(f"Pred chunks: {pred_chunks}")
            else:
                # === 句级分类 ===
                label_counter.update(labels.cpu().tolist())
                
                # 检查是否为特殊模型类型
                if hasattr(args, 'tam_cl') and args.tam_cl:
                    # TAM-CL 模型
                    try:
                        out = model(input_ids, attention_mask, token_type_ids, image_tensor, session_id=args.session_name)
                        if isinstance(out, tuple) and len(out) == 3:
                            logits, _, _ = out
                        else:
                            logits = out
                    except Exception as e:
                        logger.warning(f"TAM-CL forward failed, using fallback: {e}")
                        # 使用fallback方法
                        fused_cls = model.base_model(input_ids, attention_mask, token_type_ids, image_tensor,
                                                     return_sequence=False)
                        logits = model.head(fused_cls)
                elif hasattr(args, 'clap4clip') and args.clap4clip:
                    # CLAP4CLIP 模型
                    try:
                        logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            image_tensor=image_tensor,
                            task_name=args.session_name
                        )
                    except Exception as e:
                        logger.warning(f"CLAP4CLIP forward failed, using fallback: {e}")
                        # 使用fallback方法
                        fused_cls = model.base_model(input_ids, attention_mask, token_type_ids, image_tensor,
                                                     return_sequence=False)
                        logits = model.head(fused_cls)
                else:
                    # 标准模型
                    try:
                        fused_cls = model.base_model(input_ids, attention_mask, token_type_ids, image_tensor,
                                                     return_sequence=False)
                        if use_ddas:
                            branch_mask, _ = model.ddas(fused_cls)
                            if (~branch_mask).any():
                                plain_feat = model.base_model.base_model(
                                    input_ids[~branch_mask], attention_mask[~branch_mask],
                                    token_type_ids[~branch_mask], image_tensor[~branch_mask],
                                    return_sequence=False
                                )
                                fused_cls[~branch_mask] = plain_feat
                        
                        # 检查head类型，确保句级任务使用正确的head
                        if hasattr(model, 'head'):
                            try:
                                logits = model.head(fused_cls)  # => (b, num_labels)
                            except ValueError as e:
                                if "not enough values to unpack" in str(e):
                                    # 如果head期望3维输入但得到2维，说明head类型不匹配
                                    logger.warning(f"Head type mismatch for task {task_name}, using fallback")
                                    # 创建一个简单的线性分类器作为fallback
                                    if not hasattr(model, '_fallback_head'):
                                        model._fallback_head = nn.Linear(fused_cls.size(-1), args.num_labels).to(fused_cls.device)
                                    logits = model._fallback_head(fused_cls)
                                else:
                                    raise e
                        else:
                            raise ValueError("Model has no head")
                    except Exception as e:
                        logger.warning(f"Standard model forward failed: {e}")
                        # 如果head失败，尝试使用默认head
                        if hasattr(model, 'head'):
                            logits = model.head(fused_cls)
                        else:
                            raise e
                
                preds = torch.argmax(logits, dim=1)
                all_preds_token.extend(preds.cpu().tolist())
                all_labels_token.extend(labels.cpu().tolist())


    # === 计算评估指标 ===
    # 先过滤掉 -100
    valid_preds = []
    valid_labels = []
    for p, g in zip(all_preds_token, all_labels_token):
        if g != -100:
            valid_preds.append(p)
            valid_labels.append(g)
    
    # 计算token-level准确率（仅作参考）
    token_acc = accuracy_score(valid_labels, valid_preds)
    
    if is_sequence_task:
        # ============================================================
        # 序列标注任务 (MNER, MATE, MABSA): 使用 Chunk-level F1
        # ============================================================
        # 计算 chunk-level P/R/F1
        tp, fp, fn = 0, 0, 0
        for pset, gset in zip(all_chunks_pred, all_chunks_gold):
            tp += len(pset.intersection(gset))
            fp += len(pset - gset)
            fn += len(gset - pset)
        
        chunk_prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        chunk_rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
        chunk_f1 = 2*chunk_prec*chunk_rec/(chunk_prec+chunk_rec) if (chunk_prec+chunk_rec)>0 else 0.0
        
        # 同时计算token-level的macro指标作为参考
        token_macro_prec, token_macro_rec, token_macro_f1, _ = precision_recall_fscore_support(
            valid_labels, valid_preds, average='macro', zero_division=0
        )
        
        # 计算token-level的micro F1（排除O标签，通常label=0）
        # 过滤掉O标签（label=0）
        valid_preds_no_o = []
        valid_labels_no_o = []
        for p, g in zip(valid_preds, valid_labels):
            if g != 0:  # 排除O标签
                valid_preds_no_o.append(p)
                valid_labels_no_o.append(g)
        
        # 计算排除O标签后的micro F1
        if len(valid_labels_no_o) > 0:
            token_micro_prec, token_micro_rec, token_micro_f1, _ = precision_recall_fscore_support(
                valid_labels_no_o, valid_preds_no_o, average='micro', zero_division=0
            )
        else:
            token_micro_prec = token_micro_rec = token_micro_f1 = 0.0
        
        metrics = {
            "acc": chunk_f1 * 100.0,  # 默认主指标：Chunk F1（向后兼容）
            "chunk_precision": chunk_prec * 100.0,
            "chunk_recall": chunk_rec * 100.0,
            "chunk_f1": chunk_f1 * 100.0,  # ✨ 序列任务主指标1
            "token_acc": token_acc * 100.0,
            "token_macro_f1": token_macro_f1 * 100.0,
            "token_micro_f1_no_o": token_micro_f1 * 100.0,  # ✨ 序列任务主指标2（排除O标签的micro F1）
            "token_micro_precision_no_o": token_micro_prec * 100.0,
            "token_micro_recall_no_o": token_micro_rec * 100.0,
            "label_counter": label_counter,
        }
        logger.info(f"[{task_name}] Chunk F1: {chunk_f1*100:.2f}% (主指标1), "
                   f"Token Micro F1 (无O): {token_micro_f1*100:.2f}% (主指标2), "
                   f"Token Acc: {token_acc*100:.2f}% (参考)")
    
    else:
        # ============================================================
        # 句级分类任务 (MASC, MNRE): 使用 Micro F1
        # ============================================================
        # 计算macro平均指标
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
            valid_labels, valid_preds, average='macro', zero_division=0
        )
        
        # 计算micro平均指标（作为参考）
        micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
            valid_labels, valid_preds, average='micro', zero_division=0
        )
        
        # 计算weighted平均指标（作为参考）
        weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
            valid_labels, valid_preds, average='weighted', zero_division=0
        )
        
        metrics = {
            "acc": micro_f1 * 100.0,  # 主指标：Micro F1
            "accuracy": token_acc * 100.0,
            "macro_precision": macro_prec * 100.0,
            "macro_recall": macro_rec * 100.0,
            "macro_f1": macro_f1 * 100.0,
            "micro_f1": micro_f1 * 100.0,
            "weighted_f1": weighted_f1 * 100.0,
            "label_counter": label_counter,
        }
        logger.info(f"[{task_name}] Micro F1: {micro_f1*100:.2f}% (主指标), "
                   f"Macro F1: {macro_f1*100:.2f}% (参考), "
                   f"Accuracy: {token_acc*100:.2f}% (参考)")
    
    # 打印标签分布
    logger.info(f"Label Distribution for {task_name} {split}: {label_counter}")
    return metrics


def evaluate_all_learned_tasks(model, sessions_list, device, train_info):
    acc_list = []
    seen_sessions = set()  # 用于检测重复session
    
    logger.info(f"Evaluating {len(sessions_list)} sessions")
    for i, session in enumerate(sessions_list):
        if "task_name" not in session:
            logger.info(f"Session {session['session_name']} has no task_name, skipping")
            continue
        if "session_name" not in session:
            logger.info(f"Session at index {i} has no session_name, skipping")
            continue
            
        session_name = session["session_name"]
        tname = session["task_name"]
        
        # 检查是否有重复session
        if session_name in seen_sessions:
            logger.warning(f"Duplicate session {session_name} found at index {i}, skipping")
            continue
        seen_sessions.add(session_name)
        
        args = session["args"]
        
        # 尝试设置活动头，如果方法不存在则跳过
        try:
            if hasattr(model, 'set_active_head'):
                model.set_active_head(session["session_name"])
        except Exception as e:
            logger.warning(f"Could not set active head for session {session['session_name']}: {e}")
        
        metrics = evaluate_single_task(model, tname, "test", device, args)
        # 你可取 chunk_f1 或 accuracy 作为acc
        # if tname in ["mate", "mner", "mabsa"]:
        #     acc_list.append(m["chunk_f1"])
        # else:
        #     acc_list.append(m["accuracy"])
        logger.info(f"[Info] Evaluation metrics for {tname} on test set: {metrics['acc']}")
        acc_list.append(metrics["acc"])
    
    logger.info(f"Final acc_list: {acc_list}")
    return acc_list


