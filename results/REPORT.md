# ผลประเมินจาก corpus จริง

ตรวจ chunk coverage เพิ่มเติม: ครอบคลุมตัวอักษร 100%, ขนาดสูงสุด 1,200 อักขระ, chunk IDs ไม่ซ้ำ
Unit tests ผ่าน 10/10 ข้อ; ส่วน embedding/Gemini ใช้ mocks ไม่ใช่ live model tests

เอกสาร 4 ไฟล์, chunks 99, คำถาม 12 ข้อ

เทียบ analyzer ด้วย chunks และคำถามเดียวกัน ไม่ใช่ผลก่อน/หลังทั้ง pipeline ของต้นฉบับ

| Run | Hit@5 | Recall@5 | MRR@5 | nDCG@5 | Mean ms |
|---|---:|---:|---:|---:|---:|
| standard_bm25 | 0.000 | 0.000 | 0.000 | 0.000 | 2071.2 |
| thai_bm25 | 1.000 | 0.875 | 0.656 | 0.679 | 2071.0 |

ข้อจำกัด: ชุดพัฒนา 12 ข้อจากเอกสาร 4 ไฟล์ ใช้ source + evidence anchor กำหนด relevance อาจพลาดหลักฐานที่ใช้ถ้อยคำอื่น ยังต้องให้คนตรวจ qrels และเพิ่มชุดทดสอบที่กันไว้ต่างหาก ตัวเลขนี้ไม่ใช่ความถูกต้องของคำตอบทางการแพทย์

ยังไม่ได้วัด: answer correctness, faithfulness, citation support, unanswerable detection, contextualization quality, dense retrieval, hybrid retrieval, reranking

Latency เป็นหนึ่งรอบ รวม client overhead และอาจมี cold cache ไม่ใช้สรุปความเร็วเชิง production
