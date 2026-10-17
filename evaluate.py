"""Controlled analyzer A/B on real corpus, evidence-anchor chunk relevance.

This is a small development benchmark, not a held-out clinical evaluation.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path
from rag import ROOT, CORPUS, Search, load_chunks, digest

CASES = [
    ('r1', 'โรคหัดเยอรมันเกิดจากไวรัสอะไร', '1.md', 'Rubella virus'),
    ('r2', 'หัดเยอรมันอันตรายต่อหญิงตั้งครรภ์ช่วงไหน', '1.md', 'ไตรมาสแรก'),
    ('r3', 'วัคซีนรวมป้องกันหัดเยอรมันมีชื่อย่อว่าอะไร', '1.md', 'MMR ย่อมาจาก'),
    ('c1', 'แบคทีเรียก่ออหิวาตกโรคชื่ออะไร', '2.md', 'Vibrio cholerae'),
    ('c2', 'ซีโรกรุ๊ปใดทำให้เกิดการระบาดอหิวาตกโรค', '2.md', 'O139'),
    ('c3', 'อหิวาตกโรคมีชื่อเรียกอื่นอะไรบ้าง', '2.md', 'โรคลงราก'),
    ('e1', 'เลนส์ตาอยู่ตำแหน่งใดของตา', '44.md', 'อยู่หลังม่านตา'),
    ('e2', 'ผ่าตัดต้อกระจกด้วยคลื่นเสียงเรียกว่าอะไร', '44.md', 'Phacoemulsification'),
    ('e3', 'เลนส์ตามีเส้นผ่าศูนย์กลางเท่าไร', '44.md', '9 มิลลิเมตร'),
    ('g1', 'กรดไหลย้อนมีชื่อภาษาอังกฤษว่าอะไร', '5555.md', 'Gastroesophageal reflux disease'),
    ('g2', 'กรดไหลย้อนเกิดจากอะไร', '5555.md', 'เกิดจากภาวะหย่อนสมรรถภาพ'),
    ('g3', 'อาการของโรคกรดไหลย้อนเป็นอย่างไร', '5555.md', 'แสบร้อนกลางอก'),
]


def metrics(ids, relevant, k):
    top = ids[:k]
    gains = [int(key in relevant) for key in top]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {f'hit@{k}': int(any(gains)), f'recall@{k}': sum(gains) / len(relevant),
            f'mrr@{k}': next((1 / (i + 1) for i, g in enumerate(gains) if g), 0),
            f'ndcg@{k}': dcg / ideal}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neural', action='store_true', help='Also evaluate dense and hybrid (downloads embedding model)')
    parser.add_argument('--reranker', help='Optional multilingual cross-encoder model')
    args = parser.parse_args()
    docs = load_chunks()
    qrels = []
    for case_id, query, source, anchor in CASES:
        relevant = {d['chunk_id'] for d in docs if d['source'] == source and anchor in d['content']}
        if not relevant:
            raise ValueError(f'Missing evidence anchor: {case_id}: {anchor}')
        qrels.append({'id': case_id, 'query': query, 'source': source, 'anchor': anchor,
                      'relevant': sorted(relevant)})
    search = Search()
    report = {'scope': 'Development benchmark; real corpus; anchor-based relevance; no LLM answer evaluation',
              'source_base_commit': '4567d3d', 'files': len(list(CORPUS.glob('*.md'))),
              'chunks': len(docs), 'questions': len(qrels),
              'chunk_config': {'unit': 'unicode_characters', 'size': 1200, 'overlap': 150},
              'corpus_sha256': {p.name: digest(p.read_text(encoding='utf-8-sig')) for p in sorted(CORPUS.glob('*.md'))},
              'qrels': qrels, 'runs': {}, 'unmeasured': ['answer correctness', 'faithfulness',
                  'citation support', 'unanswerable detection', 'contextualization quality']}
    covered_total, total_chars = 0, 0
    for path in sorted(CORPUS.glob('*.md')):
        text = path.read_text(encoding='utf-8-sig')
        spans = sorted((d['start'], d['end']) for d in docs if d['source'] == path.name)
        end, covered = 0, 0
        for a, b in spans:
            covered += max(0, b - max(a, end))
            end = max(end, b)
        covered_total += covered
        total_chars += len(text)
    report['chunk_checks'] = {'character_coverage': covered_total / total_chars,
        'max_chunk_characters': max(len(d['content']) for d in docs),
        'unique_ids': len({d['chunk_id'] for d in docs}) == len(docs),
        'note': 'Coverage and bounded size do not prove semantic boundary quality.'}
    encoder, vectors = None, None
    if args.neural:
        from rag import Encoder
        encoder = Encoder()
        vectors = encoder.encode([d['content'] + '\n' + d['context'] for d in docs])
    for analyzer in ('standard', 'thai'):
        index = search.build(docs, analyzer, vectors if analyzer == 'thai' else None,
                             encoder.name if encoder else None)
        modes = ['bm25'] + (['dense', 'hybrid'] if analyzer == 'thai' and encoder else [])
        if args.reranker and analyzer == 'thai':
            modes.append('hybrid_reranked' if encoder else 'bm25_reranked')
        for mode in modes:
            rows = []
            for q in qrels:
                started = time.perf_counter()
                sparse = search.search(index, q['query'], k=20) if mode != 'dense' else []
                hits = sparse
                if mode in ('dense', 'hybrid', 'hybrid_reranked'):
                    from rag import fuse
                    dense = search.search(index, q['query'], k=20, vector=encoder.encode([q['query']])[0])
                    hits = dense if mode == 'dense' else fuse([sparse, dense])
                if mode.endswith('reranked'):
                    from rag import rerank
                    hits = rerank(q['query'], hits, args.reranker)
                ids = [d['chunk_id'] for d in hits]
                row = {'id': q['id'], 'query': q['query'], 'latency_ms': (time.perf_counter()-started)*1000,
                       'ranking': [{'chunk_id': d['chunk_id'], 'source': d['source'], 'score': d['score'],
                                    'rerank_score': d.get('rerank_score')} for d in hits]}
                for k in (1, 5, 10):
                    row.update(metrics(ids, set(q['relevant']), k))
                rows.append(row)
            summary = {key: statistics.mean(row[key] for row in rows)
                       for key in rows[0] if '@' in key}
            summary['latency_mean_ms'] = statistics.mean(row['latency_ms'] for row in rows)
            report['runs'][analyzer + '_' + mode] = {'index': index, 'summary': summary, 'cases': rows}
    if not encoder:
        report['unmeasured'].extend(['dense retrieval', 'hybrid retrieval'])
    if not args.reranker:
        report['unmeasured'].append('reranking')
    target = ROOT / 'results'
    target.mkdir(exist_ok=True)
    (target / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({name: run['summary'] for name, run in report['runs'].items()}, indent=2))
    lines = ['# ผลประเมินจาก corpus จริง', '',
             f'เอกสาร {report["files"]} ไฟล์, chunks {len(docs)}, คำถาม {len(qrels)} ข้อ', '',
             f'Character coverage: {report["chunk_checks"]["character_coverage"]:.1%}; '
             f'max chunk characters: {report["chunk_checks"]["max_chunk_characters"]}; '
             f'unique IDs: {report["chunk_checks"]["unique_ids"]}', '',
             'เทียบ analyzer ด้วย chunks และคำถามเดียวกัน ไม่ใช่ผลก่อน/หลังทั้ง pipeline ของต้นฉบับ', '',
             '| Run | Hit@5 | Recall@5 | MRR@5 | nDCG@5 | Mean ms |',
             '|---|---:|---:|---:|---:|---:|']
    for name, run in report['runs'].items():
        s = run['summary']
        lines.append(f'| {name} | {s["hit@5"]:.3f} | {s["recall@5"]:.3f} | {s["mrr@5"]:.3f} | {s["ndcg@5"]:.3f} | {s["latency_mean_ms"]:.1f} |')
    lines += ['', 'ข้อจำกัด: ชุดพัฒนา 12 ข้อจากเอกสาร 4 ไฟล์ ใช้ source + evidence anchor กำหนด relevance '
              'อาจพลาดหลักฐานที่ใช้ถ้อยคำอื่น ยังต้องให้คนตรวจ qrels '
              'และเพิ่มชุดทดสอบที่กันไว้ต่างหาก ตัวเลขนี้ไม่ใช่ความถูกต้องของคำตอบทางการแพทย์', '',
              'ยังไม่ได้วัด: ' + ', '.join(report['unmeasured']), '',
              'Latency เป็นหนึ่งรอบ รวม client overhead และอาจมี cold cache ไม่ใช้สรุปความเร็วเชิง production']
    (target / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
