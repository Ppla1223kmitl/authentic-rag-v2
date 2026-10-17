# Authentic RAG v2

Standalone repository: Thai BM25, optional Gemini/embedding/reranker, reproducible evaluation.

โค้ดชุดใหม่พัฒนาจากแนวทางใน [2025-authenticRAG](https://github.com/aekanun2020/2025-authenticRAG) base commit `4567d3d` โดยแยกไฟล์จากต้นฉบับ ไม่แก้ทับ และใช้ corpus เดิม 4 ไฟล์

สำเนา corpus อยู่ใน `corpus_input` พร้อมรันโดยไม่ต้อง clone ต้นฉบับซ้ำ
`active-index.json` เป็นสถานะ OpenSearch เฉพาะเครื่อง ไม่ขึ้น Git; รัน ingest เพื่อสร้างใหม่

## เริ่มใช้งาน (PowerShell จากโฟลเดอร์ repository)

```powershell
docker compose up -d
# ประเมิน BM25 ใช้ Python standard library ไม่ต้องมีคีย์หรือดาวน์โหลดโมเดล
python evaluate.py
python -m unittest discover -s . -p test_core.py -v

# สร้าง index ใช้งานจริง แล้วค้นหาแบบ lexical โดยยังไม่เรียก LLM
python rag.py ingest
python rag.py ask --question "กรดไหลย้อนเกิดจากอะไร"
```

ผลประเมิน: `results/REPORT.md` (สรุป), `results/evaluation.json` (qrels, อันดับ, คะแนน และผลรายคำถาม)
การประเมินสร้าง index แยก ไม่เปลี่ยน `active-index.json` ของระบบใช้งานจริง

## เปิด neural retrieval / Gemini / reranker

ใช้ virtual environment ของ Python ที่รองรับ dependencies:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# ใส่คีย์ใหม่ใน .env ด้วย editor ในเครื่อง
python rag.py ingest --neural
python rag.py ask --neural --generate --question "กรดไหลย้อนเกิดจากอะไร"

# เปิด contextualization โดยส่งเอกสารเต็มและ chunk ให้ Gemini (มีการเรียก API หลายครั้ง)
python rag.py ingest --neural --contextualize

# ประเมิน dense และ RRF ด้วยชุดเดียวกัน; ต้องดาวน์โหลด embedding model
python evaluate.py --neural
# เปิด reranker โดยระบุชื่อ multilingual cross-encoder ที่เลือก
python evaluate.py --neural --reranker YOUR_MULTILINGUAL_CROSS_ENCODER
python rag.py ask --neural --reranker YOUR_MULTILINGUAL_CROSS_ENCODER --question "กรดไหลย้อนเกิดจากอะไร"
```

ไม่มีการใช้คีย์ที่เคยส่งในแชตหรือคัดลอกคีย์จากโค้ดเดิม หากไม่มี `.env` ที่ตั้งค่าไว้ จะหยุดพร้อมแจ้งข้อผิดพลาด
Environment variables มีลำดับเหนือ `.env` เลือกรุ่น Gemini ด้วย `GEMINI_MODEL`
ค่า `RERANK_MODEL` ในตัวอย่างเป็นช่องบันทึกชื่อรุ่น; เปิดใช้งานจริงผ่าน `--reranker`

## ตัวชี้วัด

| Metric | ความหมาย | การตีความ |
|---|---|---|
| Hit@k | สัดส่วนคำถามที่มี relevant chunk อย่างน้อยหนึ่งชิ้นใน top-k | เหมาะกับการถามว่าค้นหลักฐานเจอหรือไม่ |
| Recall@k | จำนวน relevant chunks ที่เจอ / relevant chunks ทั้งหมด | วัดความครบตาม qrels; overlap อาจทำให้มีหลายชิ้นสำหรับหลักฐานเดียว |
| MRR@k | ค่าเฉลี่ย 1/อันดับของ relevant chunk แรก ถ้าไม่เจอใน k ให้ 0 | หลักฐานแรกอยู่สูงขึ้นหรือไม่ |
| nDCG@k | ความเกี่ยวข้องแบบ binary ลดน้ำหนักตามอันดับ เทียบอันดับอุดมคติ | หลักฐานหลายชิ้นเรียงดีเพียงใด |
| Latency mean | เวลาฝั่ง client ต่อคำถาม | หนึ่งรอบเท่านั้น ไม่ใช่ benchmark ความเร็ว production |
| Character coverage | สัดส่วนตัวอักษรต้นฉบับที่มีในอย่างน้อยหนึ่ง chunk | ตรวจข้อมูลตกหล่น ไม่ยืนยันความสมบูรณ์ของความหมาย |

RRF score ไม่ใช่ probability และไม่ควรเทียบตัวเลขตรงกับ BM25/cosine/reranker score

## ข้อจำกัดที่ยังต้องทดสอบ

- ชุดพัฒนา 12 ข้อ ไม่ใช่ held-out test และไม่ใช่การประเมินคำแนะนำทางการแพทย์ เอกสารมีข้อมูลเก่าตามต้นฉบับ
- qrels ใช้ชื่อไฟล์ร่วมกับข้อความหลักฐานที่กำหนดไว้ ไม่ใช่การติดป้ายความเกี่ยวข้องโดยผู้เชี่ยวชาญครบทุก chunk ต้องตรวจและเพิ่ม paraphrase, multi-hop และคำถามไม่มีคำตอบ
- ผลเทียบ `standard`/`thai` เปลี่ยน analyzer อย่างเดียวบน chunks ชุดใหม่ทั้งคู่ ไม่ใช่การรัน pipeline เดิมเต็มระบบแล้วเทียบกับ v2
- ไม่อ้างว่า embedding, reranker หรือคำตอบดีขึ้นหากไม่ได้เปิดและรันส่วนเหล่านั้น รายการ `unmeasured` อยู่ในผล JSON
- เมื่อ dense search คืนเอกสารที่ไม่เกี่ยวข้องยังไม่มี threshold ที่ปรับเทียบจาก validation set; มีเพียง no-hit abstention และ prompt ให้ปฏิเสธ จึงยังต้องประเมิน false-answer rate
- Citation validation ตรวจแค่ ID มีอยู่ใน context; ไม่ได้พิสูจน์ว่าทุกข้อความได้รับการสนับสนุน ต้องวัด faithfulness และ citation support เพิ่ม
- Chunk เป็น Unicode character windows มี overlap และเลือกขอบย่อหน้าเมื่อทำได้ อาจตัดกลางประโยคหรือ grapheme cluster; ต้องทดสอบขนาด/overlap เพิ่ม การตรวจ tokenizer ป้องกัน silent truncation โดยหยุด ไม่ได้ตัด token ให้เอง
- Index แต่ละ run ถูกเก็บไว้เพื่อ audit ต้องวางแผน retention เมื่อใช้ระยะยาว ไม่ลบข้อมูลเก่าอัตโนมัติ
- ยังไม่ใช่ระบบ production: ไม่มี authentication ของ OpenSearch, access control เอกสาร, retry/backoff, monitoring หรือ API server ใช้เฉพาะเครื่อง local ตาม docker compose

เอกสารอ้างอิง analyzer: https://docs.opensearch.org/latest/analyzers/language-analyzers/thai/
