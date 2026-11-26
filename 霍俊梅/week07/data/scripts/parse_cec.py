"""
parse_cec.py — 解析 CEC XML 语料，按 conf/event_schema.json 产出带 span 的 DuEE 格式数据集。

输入:
  data/CEC-Corpus/CEC/<event_type>/*.xml
  data/CEC-Corpus/conf/event_schema.json
  data/CEC-Corpus/conf/trigger_whitelist.json
  data/CEC-Corpus/conf/cec_mapping.json   (实体 tag/sid-oid → 角色 的固定映射)

输出 (按文档分层抽样 70/15/15; 文档级切分, 同文档样本不跨集合):
  data/CEC-Corpus/train.json   (JSONL)
  data/CEC-Corpus/dev.json
  data/CEC-Corpus/test.json

样本格式 (一条样本 = 一个核心事件; 同句多 event 拆成多条, 共享 sent_id 和 text):
  {"id":..., "sent_id":..., "text":"...",
   "event":{"type":..., "trigger":{"text":..,"start":N,"end":M},
            "args":[{"role":..,"text":..,"start":N,"end":M}]}}
  start/end 为句子内 char 偏移, 半开区间 [start, end), end - start == len(text).

Span 提取原理:
  - 遍历 <Sentence> 子树, 按 elem.text + child.tail 顺序拼接字符
  - 实体标签 (Time/Location/Participant/Object/Denoter) 进入时 start=当前字符数, 退出时 end=当前字符数
  - 实体标签外部空白字符 (\n\t 空格 全角空格) 跳过, 实体内部全部保留

核心事件筛选 (conf/trigger_whitelist.json):
  - Denoter.text 含任一白名单词 (子串匹配) 即为核心事件, 不限 Denoter subtype
  - 白名单未命中时退化: emergency Denoter → 首句首 Event (均过滤 STOP_TRIGGERS 噪音词)

论元关联 (不按 <Event> 子树归属, 改为「句内实体全收 + 来源过滤」):
  CEC 常把一个事件的论元拆到同句多个兄弟 <Event> (如 32人丧生 / 50人不适),
  且兄弟事件与触发事件之间不一定有 Causal 关系。故核心事件的论元 =
    1) 触发事件自身的实体 (全部信任)
    2) Time/Location: 触发事件没有时, 从非陈述事件的同句实体补一个
    3) Participant/Object: 同句其它事件, 排除 statement/thoughtevent 来源 (记者/警方等噪音)
       - 伤亡类事件 (denoter∈{emergency,stateChange,perception} 或含伤亡词) 里的 Participant
         一律视为伤亡 (交通的 sid→肇事者 不适用于伤亡子事件)
       - 其余按 conf/cec_mapping.json 的 (tag, sid/oid) → 角色映射
  角色最终按 conf/event_schema.json 过滤, span 仍来自 CEC 人工标注。

CLI:
  python3 parse_cec.py                        # 全量输出 (默认)
  python3 parse_cec.py --drop-empty-args      # 过滤没有任何论元的样本
  python3 parse_cec.py --drop-anaphoric       # 过滤回指触发词 ("事故发生后" 等)
"""
import os
import re
import json
import glob
import zlib
import random
import argparse
from collections import Counter
from xml.etree import ElementTree as ET
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(__file__)).parent.parent
DATA_DIR = ROOT_DIR.joinpath("data")
CORPUS_DIR = DATA_DIR.joinpath('CEC-Corpus', 'CEC')
CONF_DIR = DATA_DIR.joinpath('conf')
OUT_DIR = DATA_DIR.joinpath('processed')

SCHEMA_PATH = CONF_DIR.joinpath('event_schema.json')
WHITELIST_PATH = CONF_DIR.joinpath('trigger_whitelist.json')
MAPPING_PATH = CONF_DIR.joinpath('cec_mapping.json')
TRAIN_OUT = OUT_DIR.joinpath('train.jsonl')
DEV_OUT = OUT_DIR.joinpath('dev.jsonl')
TEST_OUT = OUT_DIR.joinpath('test.jsonl')

TYPE_MAP = {
    '地震': '地震',
    '火灾': '火灾',
    '交通事故': '交通事故',
    '恐怖袭击': '恐怖袭击',
    '食物中毒': '食物中毒',
}

ENTITY_TAGS = {'Time', 'Location', 'Participant', 'Object', 'Denoter'}
WS_CHARS = {' ', '\t', '\n', '\r', '\u3000'}

# fallback 时排除的陈述/指称类噪音词 (避免 "电/讯/报道" 等被当核心事件)
STOP_TRIGGERS = {
    '电', '讯', '报道', '说', '称', '表示', '介绍', '了解', '信息', '消息',
    '事件', '事', '记者', '发稿', '获悉', '通报', '宣布', '声明', '报告',
}

# 伤亡类事件: 其 Participant 一律视为伤亡 (见 build_args)
VICTIM_ROLES = {'伤亡人数', '受害者', '中毒者'}
CASUALTY_DENOTER_TYPES = {'emergency', 'stateChange', 'perception'}
CASUALTY_KEYWORDS = [
    '死亡', '死', '丧生', '身亡', '遇难', '受伤', '伤', '重伤', '轻伤',
    '中毒', '不适', '腹泻', '呕吐', '送医', '住院', '治疗', '病',
]


def load_schema():
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return {t['event_type']: {r['role'] for r in t['role_list']} for t in schema}


def load_trigger_whitelist():
    with open(WHITELIST_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return {et: set(words) for et, words in raw.items() if not et.startswith('_')}


def load_cec_mapping():
    """conf/cec_mapping.json: event_type -> {tag: {side: role}}"""
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return {et: m for et, m in raw.items() if not et.startswith('_')}


CEC_MAPPING = load_cec_mapping()


def validate_mapping(schema_roles):
    """确保 cec_mapping 产出的 role 都在 schema 中, 避免后续被静默过滤"""
    for et, tagmap in CEC_MAPPING.items():
        produced = set()
        for sides in tagmap.values():
            produced |= set(sides.values())
        missing = produced - schema_roles.get(et, set())
        if missing:
            raise ValueError(f'cec_mapping.json[{et}] 产出 role 不在 event_schema.json 中: {missing}')


def parse_sentence_with_spans(sentence_elem):
    """
    遍历 <Sentence>, 构建 clean text + 收集每个 <Event> 下的实体 span.

    Returns:
      sentence_text: str
      events: dict[eid] -> {
        'denoter_type': str|None, 'event_type': str|None,
        'start','end': 该 Event 在句子内的 char span,
        'trigger': {'text','start','end'}|None,
        'args': [{'tag','attrs','text','start','end','eid'}, ...]   (eid=所属事件)
      }
    """
    chars = []
    events = {}
    event_stack = []

    def append(s, preserve):
        if preserve:
            chars.extend(s)
        else:
            for ch in s:
                if ch not in WS_CHARS:
                    chars.append(ch)

    def walk(elem, in_entity):
        is_entity = elem.tag in ENTITY_TAGS
        is_event = elem.tag == 'Event'

        if is_event:
            eid = elem.get('eid')
            event_stack.append(eid)
            if eid not in events:
                events[eid] = {'denoter_type': None, 'event_type': None,
                               'trigger': None, 'args': [],
                               'start': len(chars), 'end': None}
            events[eid]['event_type'] = elem.get('type')

        ent_start = len(chars) if is_entity else None

        if elem.text:
            append(elem.text, in_entity or is_entity)

        for child in elem:
            walk(child, in_entity or is_entity)
            if child.tail:
                append(child.tail, in_entity or is_entity)

        if is_entity:
            ent_end = len(chars)
            text = ''.join(chars[ent_start:ent_end])
            cur_eid = event_stack[-1] if event_stack else None
            if cur_eid:
                if elem.tag == 'Denoter':
                    events[cur_eid]['denoter_type'] = elem.get('type')
                    events[cur_eid]['trigger'] = {'text': text, 'start': ent_start, 'end': ent_end}
                else:
                    events[cur_eid]['args'].append({
                        'tag': elem.tag,
                        'attrs': dict(elem.attrib),
                        'text': text,
                        'start': ent_start,
                        'end': ent_end,
                        'eid': cur_eid,
                    })

        if is_event:
            events[event_stack[-1]]['end'] = len(chars)
            event_stack.pop()

    walk(sentence_elem, False)
    return ''.join(chars), events


def _get_side(attrs):
    """sid/oid → 'sid'/'oid'/None; 属性名大小写不敏感 (语料里有 sID/sIDs)"""
    low = {k.lower(): v for k, v in attrs.items()}
    sid, oid = low.get('sid'), low.get('oid')
    if sid and oid:
        return 'oid'
    if sid:
        return 'sid'
    if oid:
        return 'oid'
    return None


def map_arg_role(event_type, arg):
    """按 conf/cec_mapping.json 把 (tag, sid/oid) 映射到角色; 无映射返回 None"""
    tagmap = CEC_MAPPING.get(event_type, {}).get(arg['tag'])
    if not tagmap:
        return None
    side = _get_side(arg['attrs'])
    if side in tagmap:
        return tagmap[side]
    return tagmap.get('*')


ANAPHORIC_PREFIX_CHARS = {'该', '这', '此'}


def is_anaphoric(text, trig):
    """判断触发词是否为回指 (指代前文事件, 如 '事故发生后' / '地震发生后')"""
    after = text[trig['end']:trig['end'] + 3]
    if after.startswith('发生后') or after.startswith('之后'):
        return True
    before = text[max(0, trig['start'] - 1):trig['start']]
    if before and before[-1] in ANAPHORIC_PREFIX_CHARS:
        return True
    return False


def apply_filters(samples, drop_empty_args=False, drop_anaphoric=False):
    out = samples
    if drop_empty_args:
        out = [s for s in out if s['event']['args']]
    if drop_anaphoric:
        out = [s for s in out if not is_anaphoric(s['text'], s['event']['trigger'])]
    return out


def _event_text(sent_text, ev):
    s, e = ev.get('start'), ev.get('end')
    if s is None or e is None:
        return ''
    return sent_text[s:e]


def _is_statement(ev):
    """陈述/报道事件 (记者/警方/媒体 在"说"/"报道"/"电"/"讯") — 其 Participant 是噪音"""
    return ev.get('denoter_type') == 'statement' or ev.get('event_type') == 'thoughtevent'


def _is_casualty_event(sent_text, ev):
    """伤亡类事件: 事件文本含伤亡词
    (不再仅凭 denoter_type ∈ {emergency,stateChange,perception} 判定, 以免
     存活/生存 等 stateChange 谓词的 Participant 被误当伤亡)"""
    return any(k in _event_text(sent_text, ev) for k in CASUALTY_KEYWORDS)


def build_args(event_type, sent_text, sent_events, core_eid, peers=None):
    """句内实体全收 + 来源过滤 (见模块 docstring 的「论元关联」)

    peers: 同句「触发词候选」事件 id 集合; Time/Location 只可在这些事件间互补,
           避免从无关兄弟事件 (如 市民走上街头) 借入时空。默认 {core_eid}。
    """
    if peers is None:
        peers = {core_eid}
    core_ev = sent_events[core_eid]
    seen = set()
    out = []

    def add(role, a):
        key = (role, a['start'], a['end'])
        if key in seen:
            return
        seen.add(key)
        out.append({'role': role, 'text': a['text'], 'start': a['start'], 'end': a['end']})

    # 1) 触发事件自身实体 (全部信任)
    for a in core_ev['args']:
        role = map_arg_role(event_type, a)
        if role:
            add(role, a)

    # 2) Time/Location: 触发事件没有的, 只在同句「触发词候选」(peers) 里互补一个
    for tag in ('Time', 'Location'):
        has = any(a['tag'] == tag and map_arg_role(event_type, a) for a in core_ev['args'])
        if has:
            continue
        for sid2, ev2 in sent_events.items():
            if sid2 == core_eid or sid2 not in peers or _is_statement(ev2):
                continue
            cand = next((a for a in ev2['args'] if a['tag'] == tag), None)
            if cand is not None:
                role = map_arg_role(event_type, cand)
                if role:
                    add(role, cand)
                break

    # 3) 兄弟事件: 只补「伤亡」角色 (伤亡事件里的 Participant)。
    #    肇事者/涉事车辆/袭击者 等语义角色只取触发事件自身, 避免把句子其它
    #    事件里的 警方/应急预案/收费站 等噪音误抽成虫论元。
    victim_role = '中毒者' if event_type == '食物中毒' else '伤亡人数'
    for sid2, ev2 in sent_events.items():
        if sid2 == core_eid or _is_statement(ev2):
            continue
        if not _is_casualty_event(sent_text, ev2):
            continue
        for a in ev2['args']:
            if a['tag'] == 'Participant':
                add(victim_role, a)

    return out


def pick_best_core(event_type, sent_text, sent_events, cands, peers, schema_roles=None):
    """同句多触发词候选时选一个: 角色种类多 > 覆盖地点+时间 > 论元多; 平手取最早"""
    best_eid, best_key = None, None
    for eid in cands:
        args = build_args(event_type, sent_text, sent_events, eid, peers=peers)
        if schema_roles:
            args = [a for a in args if a['role'] in schema_roles.get(event_type, set())]
        roles = {a['role'] for a in args}
        key = (len(roles), int('地点' in roles) + int('时间' in roles), len(args))
        if best_key is None or key > best_key:  # 严格大于 → 平手保留最早
            best_key, best_eid = key, eid
    return best_eid


def parse_xml_file(xml_path, event_type, schema_roles, trigger_whitelist=None):
    """返回该文档的核心事件样本列表 (每个核心 event 一条记录)"""
    if trigger_whitelist is None:
        trigger_whitelist = {}
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f'[WARN] parse failed: {xml_path}: {e}')
        return []

    root = tree.getroot()

    file_id = os.path.splitext(os.path.basename(xml_path))[0]

    # 收集所有 sentence 级别结构
    sentence_records = []  # [(sent_text, {eid: event_dict})]
    for sent in root.findall('.//Sentence'):
        sent_text, sent_events = parse_sentence_with_spans(sent)
        if not sent_text or not sent_events:
            continue
        sentence_records.append((sent_text, sent_events))

    if not sentence_records:
        return []

    # 核心事件筛选: 每句最多一条样本。
    #   触发词候选 = Denoter.text 含任一白名单词 (子串匹配, 不限 subtype)
    #   同句多候选时 pick_best_core 选覆盖最好的一个; peers = 该句全部候选
    wl = trigger_whitelist.get(event_type, set())

    def wl_candidates(sent_events):
        out = []
        for eid, ev in sent_events.items():
            trig = ev['trigger']
            if trig is None or trig['text'] in STOP_TRIGGERS:
                continue
            if any(w in trig['text'] for w in wl):
                out.append(eid)
        return out

    core_per_sentence = []  # [(sent_text, sent_events, eid, peers)]
    for sent_text, sent_events in sentence_records:
        cands = wl_candidates(sent_events)
        if not cands:
            continue
        peers = set(cands)
        best = pick_best_core(event_type, sent_text, sent_events, cands, peers, schema_roles)
        core_per_sentence.append((sent_text, sent_events, best, peers))

    # fallback: 全文无白名单命中, 退化依次尝试 emergency Denoter → 首句首 Event (均过滤噪音词)
    if not core_per_sentence:
        for sent_text, sent_events in sentence_records:
            cands = [eid for eid, ev in sent_events.items()
                     if ev['denoter_type'] == 'emergency'
                     and ev['trigger'] is not None and ev['trigger']['text'] not in STOP_TRIGGERS]
            if not cands:
                continue
            peers = set(cands)
            best = pick_best_core(event_type, sent_text, sent_events, cands, peers, schema_roles)
            core_per_sentence.append((sent_text, sent_events, best, peers))
    if not core_per_sentence:
        sent_text, sent_events = sentence_records[0]
        first_eid = next(iter(sent_events), None)
        if first_eid:
            trig = sent_events[first_eid]['trigger']
            if trig is not None and trig['text'] not in STOP_TRIGGERS:
                core_per_sentence.append((sent_text, sent_events, first_eid, {first_eid}))

    samples = []
    for idx, (sent_text, sent_events, eid, peers) in enumerate(core_per_sentence):
        ev = sent_events[eid]
        trigger = ev['trigger']
        if trigger is None:
            continue

        # 句内实体全收 + 来源过滤 (Time/Location 只在 peers 间互补; 伤亡从同句 casualty 兄弟事件补)
        arg_list = build_args(event_type, sent_text, sent_events, eid, peers=peers)

        # schema 校验
        arg_list = [a for a in arg_list if a['role'] in schema_roles.get(event_type, set())]

        samples.append({
            'id': f'{event_type}_{file_id}_{idx}',
            'sent_id': f'{event_type}_{file_id}',
            'text': sent_text,
            'event': {
                'type': event_type,
                'trigger': trigger,
                'args': arg_list,
            },
        })
    return samples


def split_docs_by_type(xml_files, event_type, ratios=(0.7, 0.15, 0.15), seed=42):
    # 用 crc32 稳定哈希: str.__hash__ 每次进程随机化, 会导致切分不可复现
    stable = zlib.crc32(event_type.encode('utf-8'))
    rng = random.Random(seed + stable)
    files = list(xml_files)
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_dev = int(n * ratios[1])
    return files[:n_train], files[n_train:n_train + n_dev], files[n_train + n_dev:]


def write_jsonl(path, samples):
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'Wrote {path}  ({len(samples)} samples)')
    
def write_json(path, samples):
    with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(samples, ensure_ascii=False) + '\n')
    print(f'Wrote {path}  ({len(samples)} samples)')


def main():
    ap = argparse.ArgumentParser(description='解析 CEC XML → DuEE 格式数据集')
    ap.add_argument('--drop-empty-args', action='store_true',
                    help='过滤没有任何论元的样本')
    ap.add_argument('--drop-anaphoric', action='store_true',
                    help='过滤回指触发词样本 (如 "事故发生后")')
    args = ap.parse_args()

    schema_roles = load_schema()
    validate_mapping(schema_roles)
    trigger_whitelist = load_trigger_whitelist()
    train, dev, test = [], [], []
    stats = Counter()

    for sub_dir, event_type in TYPE_MAP.items():
        xml_dir = os.path.join(CORPUS_DIR, sub_dir)
        if not os.path.isdir(xml_dir):
            print(f'[WARN] missing dir: {xml_dir}')
            continue
        xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
        tr_docs, dv_docs, te_docs = split_docs_by_type(xml_files, event_type)

        cnt = {'train': 0, 'dev': 0, 'test': 0}
        for docs, bucket, key in [(tr_docs, train, 'train'),
                                   (dv_docs, dev, 'dev'),
                                   (te_docs, test, 'test')]:
            for p in docs:
                s = parse_xml_file(p, event_type, schema_roles, trigger_whitelist)
                s = apply_filters(s, args.drop_empty_args, args.drop_anaphoric)
                bucket.extend(s)
                cnt[key] += len(s)
        print(f'{event_type}: docs {len(xml_files)} -> '
              f'{len(tr_docs)}/{len(dv_docs)}/{len(te_docs)}; '
              f'samples {cnt["train"]}/{cnt["dev"]}/{cnt["test"]}')
        stats[event_type] = sum(cnt.values())

    rng = random.Random(42)
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    print(f'\nTotal samples: {sum(stats.values())}  by type: {dict(stats)}')
    print(f'Train: {len(train)}  Dev: {len(dev)}  Test: {len(test)}')

    write_jsonl(TRAIN_OUT, train)
    write_jsonl(DEV_OUT, dev)
    write_jsonl(TEST_OUT, test)


if __name__ == '__main__':
    main()
