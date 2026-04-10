# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Tiến Thắng
**Nhóm:** B6-C401
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector hướng về gần như cùng một phía trong không gian vector, biểu thị rằng hai đoạn văn bản có sự tương đồng rất lớn về mặt ngữ nghĩa hoặc chủ đề.

**Ví dụ HIGH similarity:**
- Sentence A: "Hôm nay trời nắng rất đẹp và rực rỡ."
- Sentence B: "Thời tiết bên ngoài đang tràn ngập ánh nắng và rất trong xanh."
- Tại sao tương đồng: Cả hai câu đều diễn đạt cùng một trạng thái thời tiết nắng ấm, dù sử dụng cấu trúc câu và từ ngữ khác nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Tôi rất thích thưởng thức món pizza hải sản vào cuối tuần."
- Sentence B: "Thị trường tài chính toàn cầu đang biến động mạnh do lạm phát."
- Tại sao khác: Một câu thuộc chủ đề ẩm thực giải trí, câu còn lại thuộc chủ đề kinh tế vĩ mô, không có sự giao thoa về ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Vì Cosine similarity chỉ quan tâm đến hướng (chủ đề) của vector mà không bị ảnh hưởng bởi độ dài (độ lớn vector). Điều này giúp việc so sánh văn bản chính xác hơn, không bị thiên kiến khi một văn bản dài hơn văn bản kia dù cùng nội dung.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Áp dụng công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)`
> *Đáp án:* 23

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Số lượng chunk sẽ tăng lên (khoảng 25). Việc tăng overlap giúp bảo toàn ngữ cảnh tốt hơn khi các câu bị chia cắt ở ranh giới chunk, giúp LLM hiểu được mối liên hệ giữa các đoạn văn kế tiếp nhau.
---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Vietnamese rappers

**Tại sao nhóm chọn domain này?**
> Vì mỗi rapper có một tiểu sử, sự nghiệp, phong cách rap, và các mối quan hệ (bạn bè, kẻ thù) khác nhau, tạo ra sự đa dạng về nội dung và cấu trúc dữ liệu. Có những fact nhỏ về các rapper sẽ dễ bị lẫn mất trong quá trình embedding nếu không chunk hợp lý.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | suboi.md | RapViet Wiki | 6456 | |
| 2 | mc_ill.md | RapViet Wiki | 4823 | |
| 3 | karik.md | RapViet Wiki | 3868 | |
| 4 | b_ray.md | RapViet Wiki | 3758 | |
| 5 | rhymastic.md | RapViet Wiki | 3763 | |
| 6 | young_h.md | RapViet Wiki | 3342 | |
| 7 | phuc_du.md | RapViet Wiki | 2727 | |
| 8 | icd.md | RapViet Wiki | 2328 | |
| 9 | blacka.md | RapViet Wiki | 2198 | |
| 10 | wowy.md | RapViet Wiki | 1520 | |
| 11 | de_choat.md | RapViet Wiki | 1206 | |
| 12 | minh_lai.md | RapViet Wiki | 1158 | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| suboi.md | FixedSizeChunker (`fixed_size`) | 11 | 495.9 | Medium |
| suboi.md | SentenceChunker (`by_sentences`) | 9 | 581.0 | High |
| suboi.md | RecursiveChunker (`recursive`) | 14 | 375.4 | Highest |
| minh_lai.md | FixedSizeChunker (`fixed_size`) | 2 | 483.0 | Low |
| minh_lai.md | SentenceChunker (`by_sentences`) | 3 | 314.3 | High |
| minh_lai.md | RecursiveChunker (`recursive`) | 3 | 315.3 | High |

### Strategy Của Tôi

**Loại:** RecursiveChunker (chunk_size=150)

**Mô tả cách hoạt động:**
> Đây là chiến lược chia nhỏ văn bản đệ quy dựa trên danh sách các ký tự phân tách ưu tiên: `\n\n`, `\n`, `. `, ` ` và cuối cùng là ký tự trống. Với kích thước chunk nhỏ (150 ký tự), thuật toán sẽ cố gắng giữ nguyên các đoạn văn hoặc câu hoàn chỉnh nếu chúng vừa vặn, ngược lại sẽ lùi xuống cấp độ từ hoặc ký tự để đảm bảo không vượt quá giới hạn. Cách tiếp cận này giúp cô lập các "fact" nhỏ nhưng quan trọng thành các đơn vị tìm kiếm riêng biệt.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Domain rapper có rất nhiều thông tin nhỏ lẻ như tên thật, ngày sinh, sự kiện ẩu đả hay mâu thuẫn (beef) thường chỉ nằm trong 1-2 câu ngắn. Nếu dùng chunk quá lớn (500), các thông tin này bị pha loãng bởi các đoạn tiểu sử dài, làm giảm độ chính xác khi tìm kiếm. Với chunk_size=150, hệ thống có thể truy xuất chính xác đoạn văn chứa sự kiện cụ thể mà người dùng hỏi.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| suboi.md | Recursive (500) | 14 | 375.4 | Medium (Chứa nhiều noise) |
| suboi.md | **Recursive (150)** | 40 | 131.4 | **Highest** (Rất cô đọng) |
| minh_lai.md | Recursive (500) | 3 | 315.3 | Medium |
| minh_lai.md | **Recursive (150)** | 8 | 118.2 | **High** (Dễ trích dẫn fact) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng thư viện `re.split` với kỹ thuật **lookbehind** `(?<=\. |\! |\? |\.\n)` để chia văn bản thành các câu mà không làm mất các dấu câu kết thúc. Sau đó, các câu này được nhóm lại theo tham số `max_sentences_per_chunk` trước khi được join lại thành một chuỗi duy nhất cho mỗi chunk.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Triển khai theo giải thuật đệ quy: thử chia một khối văn bản bằng danh sách các ký tự phân tách theo thứ tự ưu tiên (`\n\n`, `\n`,...). Nếu một đoạn nhỏ vẫn vượt quá `chunk_size`, nó sẽ tiếp tục được chia đệ quy bằng ký tự phân tách tiếp theo. Cuối cùng, có một bước gộp các mảnh nhỏ (merge) lại với nhau sao cho độ dài không vượt quá giới hạn, giúp tối ưu hóa dung lượng mỗi chunk.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Lưu trữ văn bản dưới dạng danh sách các từ điển (dictionary) chứa content, embedding, và metadata. Hàm `search` thực hiện tính toán tích vô hướng (dot product) giữa embedding của câu truy vấn và toàn bộ kho lưu trữ, sau đó sắp xếp theo thứ tự giảm dần để lấy top k.

**`search_with_filter` + `delete_document`** — approach:
> Thực hiện **pre-filtering**: lọc các bản ghi thoả mãn điều kiện metadata trước, sau đó mới tiến hành tính toán độ tương đồng trên tập con đã lọc. Điều này giúp tăng hiệu năng và độ chính xác. Hàm xóa sử dụng list comprehension để loại bỏ các bản ghi có `doc_id` tương ứng.

### KnowledgeBaseAgent

**`answer`** — approach:
> Thực hiện đúng quy trình RAG: Lấy Top-k chunk liên quan nhất từ Store -> Gộp các chunk này thành một khối context lớn -> Tiêm context và câu hỏi vào một template prompt -> Gọi LLM để sinh câu trả lời dựa trên context đó.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
...
============================= 42 passed in 0.12s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Hôm nay trời nắng đẹp. | Thời tiết hôm nay rất tốt. | High | 0.6576 | Đúng |
| 2 | Tôi thích ăn pizza. | Con mèo đang ngủ trên ghế. | Low | 0.3175 | Đúng |
| 3 | Suboi là rapper. | Trang Anh là nghệ sĩ hiphop. | High | 0.3727 | Thấp hơn dự kiến |
| 4 | Hà Nội là thủ đô Việt Nam. | Paris là thủ đô nước Pháp. | Low | 0.5381 | Cao hơn dự kiến |
| 5 | ICD là quán quân KOR. | Dế Choắt thắng Rap Việt. | High | 0.203 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp câu số 4 (hai thủ đô khác nhau) có điểm số cao bất ngờ (0.53), cho thấy mô hình embedding nhận diện các cấu trúc câu tương đồng và thực thể cùng loại (Thủ đô - Quốc gia) rất mạnh. Ngược lại, cặp câu 5 về hai quán quân rap Việt lại có điểm rất thấp, cho thấy embedding tập trung nhiều vào từ khoá cụ thể hơn là khái niệm "chiến thắng một cuộc thi rap" nếu các từ khoá không trùng nhau.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Những rapper nào đã từng là kẻ thù của ICD? | Các rapper mâu thuẫn gồm: B2C, Sol'Bass, Hades, Locoboiz, Choi, Rick, Dabee, Hale, D Joker... |
| 2 | Giới thiệu về Quán quân mùa 1 của chương trình Rap Việt. | Dế Choắt (Châu Hải Minh). |
| 3 | Rapper từng rap cho Obama khi ông đến VN? | Suboi (Hàng Lâm Trang Anh). |
| 4 | Những ai là người từng ẩu đả với rapper Blacka? | Young H và B Ray (năm 2016). |
| 5 | Giới thiệu về một rapper từng học Đại Học Kiến Trúc HN. | Rhymastic (Vũ Đức Thiện). |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Những kẻ thù của ICD? | icd.md: Sol'Bass, Dabee, MC ILL, D Joker... | 0.6894 | Yes | Trình bày đúng danh sách. |
| 2 | Quán quân Rap Việt mùa 1? | de_choat.md: trở thành quán quân mùa 1... | 0.7337 | Yes | Châu Hải Minh (Dế Choắt). |
| 3 | Rapper rap cho Obama? | suboi.md: Cô từng rap cho Cựu Tổng thống... | 0.5964 | Yes | Xác nhận là Suboi. |
| 4 | Ẩu đả với Blacka? | blacka.md: từng ẩu đả với Young H và B Ray... | 0.5547 | Yes | Young H và B Ray. |
| 5 | Rapper học Kiến Trúc HN? | rhymastic.md: Rhymastic từng tốt nghiệp... | 0.5799 | Yes | Rhymastic. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ tập trung hơn vào việc xử lý các bảng (tables) trong Markdown. Hiện tại khi chia nhỏ bằng ký tự, các bảng thông tin rapper thường bị nát ra, làm mất đi sự liên kết giữa các thuộc tính. Việc sử dụng một bộ "Markdown Splitter" chuyên dụng hoặc giữ kích thước chunk đủ lớn để chứa trọn một bảng metadata sẽ giúp kết quả truy xuất ổn định hơn.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
