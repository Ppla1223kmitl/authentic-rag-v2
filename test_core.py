import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from rag import chunks, fuse, Search, Gemini, Encoder, load_chunks
from evaluate import metrics, CASES


class CoreTests(unittest.TestCase):
    def test_chunk_coverage_offsets_and_stable_ids(self):
        text = '# หัวข้อ\n\n' + 'ข้อความภาษาไทย ไม่มีเว้นวรรคบางส่วน\n\n' * 140
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'thai.md'
            path.write_text(text, encoding='utf-8')
            docs = list(chunks(path))
            self.assertEqual(docs, list(chunks(path)))
            covered = set()
            for doc in docs:
                self.assertLessEqual(len(doc['content']), 1200)
                self.assertEqual(doc['content'], text[doc['start']:doc['end']])
                covered.update(range(doc['start'], doc['end']))
            self.assertEqual(len(covered), len(text))
            old_ids = {d['chunk_id'] for d in docs}
            path.write_text(text + 'แก้ไข', encoding='utf-8')
            self.assertFalse(old_ids & {d['chunk_id'] for d in chunks(path)})

    def test_rrf_one_based_and_duplicate_protection(self):
        a, b = {'chunk_id': 'a'}, {'chunk_id': 'b'}
        result = fuse([[a, a, b], [b]])
        self.assertEqual(result[0]['chunk_id'], 'b')
        self.assertAlmostEqual(result[1]['score'], 1/61)
        with self.assertRaises(ValueError):
            fuse([[a]], 0)

    def test_metrics_known_ranking(self):
        m = metrics(['x', 'a', 'b'], {'a', 'b'}, 3)
        self.assertEqual(m['recall@3'], 1)
        self.assertEqual(m['mrr@3'], .5)
        self.assertAlmostEqual(m['ndcg@3'], (1/math.log2(3)+.5)/(1+1/math.log2(3)))
        self.assertEqual(metrics([], {'a'}, 5)['hit@5'], 0)

    def test_bulk_partial_failure_is_fatal(self):
        search = Search()
        with patch.object(search, 'call', side_effect=[{}, {'errors': True, 'items': [
                {'index': {'status': 400, 'error': 'bad vector'}}]}]):
            with self.assertRaises(RuntimeError):
                search.build([{'chunk_id': 'a', 'content': 'ข้อความ'}])

    def test_count_mismatch_is_fatal(self):
        search = Search()
        with patch.object(search, 'call', side_effect=[{}, {'errors': False, 'items': [
                {'index': {'status': 201}}]}, {}, {'count': 0}]):
            with self.assertRaises(RuntimeError):
                search.build([{'chunk_id': 'a', 'content': 'ข้อความ'}])

    def test_empty_retrieval_abstains_without_api(self):
        client = object.__new__(Gemini)
        self.assertEqual(client.answer('คำถาม', [])['citations'], [])

    def test_all_evaluation_anchors_exist(self):
        docs = load_chunks()
        for case, query, source, anchor in CASES:
            self.assertTrue(any(d['source'] == source and anchor in d['content'] for d in docs), case)

    def test_encoder_rejects_truncation_and_invalid_vectors(self):
        encoder = object.__new__(Encoder)
        encoder.model = Mock()
        encoder.model.max_seq_length = 3
        encoder.model.tokenizer.return_value = {'input_ids': [1, 2, 3, 4]}
        with self.assertRaises(ValueError):
            encoder.encode(['long input'])
        encoder.model.tokenizer.return_value = {'input_ids': [1]}
        encoder.model.get_sentence_embedding_dimension.return_value = 2
        for vector in ([0, 0], [float('nan'), 1], [1]):
            encoder.model.encode.return_value.tolist.return_value = [vector]
            with self.assertRaises(ValueError):
                encoder.encode(['input'])

    def test_answer_rejects_hallucinated_citation(self):
        client = object.__new__(Gemini)
        client.client = Mock()
        client.model = 'test'
        client.client.models.count_tokens.return_value.total_tokens = 10
        client.generate = Mock(return_value='คำตอบ [' + 'b'*64 + ']')
        with self.assertRaises(RuntimeError):
            client.answer('คำถาม', [{'chunk_id': 'a'*64, 'source': '1.md', 'content': 'หลักฐาน'}])

    def test_contextualization_uses_full_document(self):
        client = object.__new__(Gemini)
        client.client = Mock()
        client.model = 'test'
        client.client.models.count_tokens.return_value.total_tokens = 10
        client.generate = Mock(return_value='บริบท')
        doc = load_chunks()[0]
        client.contextualize([doc])
        prompt = client.generate.call_args.args[0]
        from rag import CORPUS
        self.assertIn((CORPUS / doc['source']).read_text(encoding='utf-8-sig'), prompt)
        self.assertEqual(doc['context'], 'บริบท')


if __name__ == '__main__':
    unittest.main()
