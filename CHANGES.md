# บันทึกการแก้ไขจากโค้ดต้นฉบับ

จัดทำ 17 ตุลาคม 2026 — ต้นทาง aekanun2020/2025-authenticRAG commit 4567d3d

| ขั้นตอน | ปัญหาจากต้นฉบับ | การแก้ใน v2 | หลักฐาน/วิธีวัด |
|---|---|---|---|
| Loading | path ผูกกับ working directory และ encoding ไม่ระบุ | UTF-8-sig, path อิงตำแหน่งสคริปต์, ตรวจ corpus ว่าง | รันจาก workspace ได้, corpus hash ใน JSON |
| Chunking | ตั้ง `chunk_size` แต่ไม่มีการตรวจ chunks จริง | ช่วงอักขระขนาดจำกัด มี overlap, หัวข้อ, source และ offsets | unit test coverage/offset/size และ chunk_checks |
| Contextualization | ส่ง chunk เป็นทั้งเอกสารเต็มและ chunk | โหมดเลือกใช้ส่งเอกสารเต็มจริง ตรวจ token budget; ค่าเริ่มต้นใช้หัวข้อไม่เสีย API | โค้ดเส้นทางชัดเจน; ยังไม่ได้วัดคุณภาพ Gemini |
| Embedding | ไม่ตรวจ silent truncation หรือค่า vector | tokenizer ของโมเดลจริง, normalize, ตรวจมิติ/finite/zero; ใช้รุ่น index เดียวกับ query | มี validation; ยังไม่ได้รัน neural ในสภาพแวดล้อมนี้ |
| Indexing | ID ตามลำดับเขียนทับ/ข้อมูลเก่าค้าง | ID จาก source/document hash/offset/config; run index ใหม่ และ publish manifest หลังตรวจสำเร็จ | unit test stable/versioned IDs และ count mismatch |
| Metadata | source/context สูญหายจากผล vector | เก็บ content/context/source/heading/offset/model ใน index เดียว | BM25 และ vector ใช้ chunk IDs ชุดเดียวกัน |
| Bulk | error แค่พิมพ์แจ้งและคืนจำนวนทั้งหมด | ตรวจทุก item, หยุดเมื่อ fail, ตรวจ count หลัง refresh | partial failure และ count mismatch tests |
| Analyzer | standard จับคำไทยไม่เหมาะกับ corpus | thai ทั้ง index/search time | controlled A/B ใน results |
| RRF | ใช้อันดับเริ่ม 0 | ใช้อันดับเริ่ม 1, deduplicate, tie-break คงที่ | สูตรและ unit test; ไม่อ้างว่าทำให้ relevance ดีขึ้นเอง |
| Candidates | ผูก candidate count กับ output top-k | ดึง 20 candidates ต่อ branch ก่อน fuse/rerank แล้วเลือก 5 | trace ในผล ask และ evaluation |
| Reranking | ไม่มีขั้น reranker | เปิดใช้ multilingual CrossEncoder ผ่าน flag; cache รุ่นโมเดล | ยังไม่มีผลที่วัดหากไม่ได้ระบุรุ่นและรัน |
| Context budget | ใช้ split คำประมาณ token | Gemini count_tokens กับ prompt รวมคำถาม/หลักฐานจริง เลือกเฉพาะ chunk ที่ใส่ได้ | implemented; live Gemini ยังไม่วัด |
| Generation | ไม่มี citation, อาจตอบเองเมื่อค้นไม่เจอ | no-hit abstention, prompt ยึดหลักฐาน, temperature 0, ตรวจ citation IDs | unit test no-hit; ไม่อ้างว่าป้องกัน hallucination ได้ทั้งหมด |
| API errors | error string อาจเข้า corpus | exception ทำให้หยุด; ไม่ publish index ที่ยังสร้างไม่ครบ | partial failure test; API error path ไม่คืนข้อความ error เป็นเนื้อหา |
| Evaluation | ไม่มี qrels และ metric แยก stage | 12 queries + evidence anchors จาก 4 ไฟล์; บันทึกอันดับ คะแนน hashes และ metrics | REPORT.md + evaluation.json |

ผลจริงและข้อจำกัดอ่านใน results/REPORT.md การแก้เชิงโค้ดกับผลเชิงคุณภาพเป็นคนละข้อกล่าวอ้าง ส่วนที่ยังไม่ได้รันทดสอบจริงระบุว่า unmeasured แทนการใส่ตัวเลขสมมติ

## ผลที่วัดได้จริง

- Corpus เดิม 4 ไฟล์ ได้ 99 chunks ครอบคลุมตัวอักษร 100%, ขนาดสูงสุด 1,200 อักขระ, IDs ไม่ซ้ำ
- Unit tests ผ่าน 10/10 ข้อ (รวม mocked embedding/citation/contextualization; ไม่ใช่ live neural/API test)
- คำถาม 12 ข้อ: standard Hit@5 = 0%, thai Hit@5 = 100%
- Thai Hit@1 = 50%, Recall@5 = 87.5%, MRR@5 = 0.6556, nDCG@5 = 0.6786
- ยังมีปัญหาอันดับ: หลักฐานอยู่ที่อันดับแรกเพียง 6/12 คำถาม จึงไม่อ้างว่า retrieval สมบูรณ์ และไม่อ้างว่าคำตอบถูกต้อง 100%
- Dense/hybrid/reranker และ Gemini ยังไม่ได้วัดสด: ไม่มี sentence-transformers ใน environment ที่ตรวจ จึงจัดเตรียมทางเลือกสำหรับรันต่อโดยไม่สร้างตัวเลขแทน
