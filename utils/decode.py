# utils/decode.py

def decode_mate(label_ids):
    """
    解码 MATE 的序列标签(单个句子). O=0, B=1, I=2
    返回: set of (start, end), 表示抽取到的方面词位置区间.
    label_ids: List[int], 已过滤掉 -100（或在外部先过滤）
    """
    # 确保 label_ids 是扁平化的整数列表
    if isinstance(label_ids, list) and len(label_ids) > 0 and isinstance(label_ids[0], list):
        # 如果是嵌套列表，展平它
        label_ids = [item for sublist in label_ids for item in sublist]
    
    # 确保所有元素都是整数
    label_ids = [int(label) if isinstance(label, (int, float)) else label for label in label_ids]
    
    aspects = set()
    start = None
    for i, label in enumerate(label_ids):
        if label == 1:  # B
            # 若之前有未闭合块 => 先闭合(容错)
            if start is not None:
                aspects.add((start, i-1))
            # 新的块开始
            start = i
        elif label == 2:  # I
            # 连续
            pass
        else:  # O=0
            # 结束之前B/I块
            if start is not None:
                aspects.add((start, i-1))
                start = None

    # 若最后还没闭合
    if start is not None:
        aspects.add((start, len(label_ids)-1))

    return aspects


def decode_mner(label_ids):
    """
    MNER 解码 (单一句子, 不含 -100).
    label_ids: List[int] in {0,1..8}
      O=0,
      B-type = 1 + 2*t
      I-type = 2 + 2*t
    其中 t in [0..3], 共4个实体类型 => 对应9类(含O=0).
    返回: set of (start, end, t)
      其中 t 是实体类型编号 (0..3)
    """
    # 确保 label_ids 是扁平化的整数列表
    if isinstance(label_ids, list) and len(label_ids) > 0 and isinstance(label_ids[0], list):
        # 如果是嵌套列表，展平它
        label_ids = [item for sublist in label_ids for item in sublist]
    
    # 确保所有元素都是整数
    label_ids = [int(lab) if isinstance(lab, (int, float)) else lab for lab in label_ids]
    
    chunks = set()
    start = None
    current_type = None

    for i, lab in enumerate(label_ids):
        if lab == 0:
            # O
            if start is not None:
                # 闭合
                chunks.add((start, i - 1, current_type))
                start, current_type = None, None
        else:
            # 要么 B-type, 要么 I-type
            # 如果是 B => (lab - 1) % 2 == 0; 如果是 I => (lab - 2) % 2 == 0
            # 也可以直接判断 lab % 2
            if lab % 2 == 1:
                # B
                # 先闭合之前的
                if start is not None:
                    chunks.add((start, i - 1, current_type))
                start_type = (lab - 1) // 2  # = t
                start = i
                current_type = start_type
            else:
                # I
                i_type = (lab - 2) // 2
                if current_type != i_type:
                    # 出现不匹配, 强行闭合
                    if start is not None:
                        chunks.add((start, i - 1, current_type))
                    start = i
                    current_type = i_type
                # 否则就连续
    # 收尾
    if start is not None:
        chunks.add((start, len(label_ids) - 1, current_type))

    return chunks


def decode_mabsa(label_ids):
    """
    MABSA 解码 (单句话).
    label_ids: List[int], e.g. [0,1,2,0,5,6,...], 不含 -100
      0=O, 1=B-neg,2=I-neg,3=B-neu,4=I-neu,5=B-pos,6=I-pos
    返回: set of (start, end, sentiment)
      sentiment in {-1,0,1} 分别表示neg/neu/pos
    """
    # 确保 label_ids 是扁平化的整数列表
    if isinstance(label_ids, list) and len(label_ids) > 0 and isinstance(label_ids[0], list):
        # 如果是嵌套列表，展平它
        label_ids = [item for sublist in label_ids for item in sublist]
    
    # 确保所有元素都是整数
    label_ids = [int(lab) if isinstance(lab, (int, float)) else lab for lab in label_ids]
    
    aspects = set()
    start = None
    current_sent = None

    def label2sent(lab):
        # 1 or 2 => neg=-1
        # 3 or 4 => neu=0
        # 5 or 6 => pos=1
        if lab in [1,2]:
            return -1
        elif lab in [3,4]:
            return 0
        elif lab in [5,6]:
            return 1
        return None

    for i, lab in enumerate(label_ids):
        if lab == 0:
            # O
            if start is not None:
                # 闭合
                aspects.add((start, i-1, current_sent))
                start, current_sent = None, None
        else:
            # B-xxx or I-xxx
            # B => lab in [1,3,5], I => [2,4,6]
            is_b = (lab % 2 == 1)  # 1,3,5 => B
            sent = label2sent(lab)
            if is_b:
                # 闭合旧的
                if start is not None:
                    aspects.add((start, i-1, current_sent))
                start = i
                current_sent = sent
            else:
                # I
                # 若和旧sent不一致, 强行闭合
                if current_sent != sent:
                    if start is not None:
                        aspects.add((start, i-1, current_sent))
                    start = i
                    current_sent = sent

    if start is not None:
        aspects.add((start, len(label_ids)-1, current_sent))

    return aspects