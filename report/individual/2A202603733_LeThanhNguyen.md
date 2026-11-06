# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603733
- **Họ và tên:** (chỉ ghi ở tên file theo quy định RULES.md)
- **SEED** (in ở ô đầu notebook): 5465

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra (15/15 ô kiểm tra đạt) | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ (cần tự chụp ảnh viết tay) |

## 2. Kết quả chính

Chạy cục bộ trên Windows (CPU, không phải Colab), dữ liệu SMS Spam Collection (5572 dòng, còn 5169 sau khi bỏ trùng). LLM chỉ chạy trên tập SUB 40 tin.
LLM dùng `gemini-3.5-flash-lite` (key Gemini miễn phí). Hàng 4b (model lớn) **không có số liệu**: gemini-3.5-flash bị giới hạn 20 request/ngày ở gói miễn phí (lỗi 429) và gemini-3.8-flash bị quá tải (503).

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.876 | 0.889 | 0.033 | 0 |
| 2. BoW + MLP | 0.944 | 0.974 | 0.160 | 0 |
| 3a. Encoder MiniLM + kNN | 0.914 | 0.950 | 24.4 | 0 |
| 3b. Encoder + LogReg | 0.950 | 0.974 | 14.7 | 0 |
| 3c. Encoder + MLP head | 0.963 | 0.974 | 15.7 | 0 |
| 4a. LLM nhỏ zero-shot | 0.976 (SUB) | 0.976 | 1295 | 0.027 |
| 4c. LLM nhỏ few-shot | 0.976 (SUB) | 0.976 | 1194 | 0.065 |
| 4b. LLM lớn zero-shot | không chạy được | — | — | — |

(Hàng LLM chỉ đo trên SUB nên cột F1 test lấy bằng F1 SUB. SUB chỉ 40 tin: chênh dưới 0.05 coi như hoà.)

## 3. Dự đoán và thực tế

1. **`dl_curve` / MLP so với LogReg.** Đoán MLP xấp xỉ LogReg (chênh khoảng 0.03). Thực tế MLP hơn rõ: F1 0.944 so với 0.876. Thí nghiệm bỏ ReLU vẫn đạt 0.944, nên chênh lệch chủ yếu đến từ tối ưu hoá (Adam, mini-batch) chứ không phải phi tuyến; LogReg ở Phần 1 chưa hội tụ sau 300 epoch (recall 0.779).
2. **`ml_lr` / lr = 50.** Đoán loss dao động dữ dội. Thực tế loss vọt lên 2.57 ở epoch đầu rồi tụt về 0.006, F1 test 0.930, cao hơn lr = 0.5 (0.876). Lệch vì bão hoà sigmoid với trọng số lớn làm loss giả thấp; đó chưa phải overfit nặng.
3. **`hard_eval`.** Đoán LLM yếu ở prompt injection. Thực tế LLM nhỏ đạt 100% cả 7 nhóm (ở cả prompt thường và prompt phòng thủ), còn BoW chỉ 52–56%. Lệch vì model hiện đại đọc hiểu ý định chứ không chỉ khớp từ khoá.

## 4. Phân tích 50 case khó

- Tụt nhiều nhất: BoW + LogReg (F1 0.889 → 0.333) và BoW + MLP (0.974 → 0.421). Embedding tụt vừa (kNN 0.704, 3b/3c 0.612). LLM không tụt (1.000).
- Nhóm khó với BoW: Thao túng (12–38%), Spam giọng tự nhiên (10%), Né từ khoá (BoW + MLP 0/6).
- Case 1 — #29 "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd…" (né từ khoá): BoW + MLP sai cả 6 case nhóm này vì các token trộn số–chữ như `c0ngr4tulat10ns` không có trong vocab 2000 từ, nên vector gần như toàn 0 và mô hình mặc định đoán ham.
- Case 2 — #30 "F R E E  e n t r y … t x t  WIN …" (né từ khoá): tách bằng dấu cách khiến mỗi chữ cái thành một token riêng (`f`, `r`, `e`...), mất hoàn toàn token `free`, `entry`, `txt`.
- Embedding kém ở nhóm Nói về spam (kNN 25%): vector nghĩa gần chủ đề tiền/giải thưởng nên tin chê/kể lại spam bị coi là spam.

## 5. Thử thách cá nhân

Đề #6: prompt trả JSON `{"label", "reason"}` thay cho 1 từ, chạy trên 10 tin SUB (5 spam + 5 ham) với model nhỏ. Kết quả: accuracy 0.90, F1 0.91 giống bản 1 từ, nhưng chi phí gấp 2.9 lần (0.0772 so với 0.0268 USD/1k tin), token output ≈ 21/tin so với ≈ 1, độ trễ 1.22 s so với 0.73 s. Không đáng khi chỉ cần nhãn; chỉ đáng khi cần giải thích/kiểm toán, và nên chỉ gọi cho tin lưỡng lự. Mẫu 10 tin chỉ đủ kết luận về chi phí, không về độ chính xác.

## 6. Phương án cho sản phẩm thật

Yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày (≈ 116 tin/s). Tôi chọn Encoder + MLP head (3c) làm lớp chính: p95 15.7 ms, F1 test 0.963, và trên 50 case khó đạt 0.612 so với 0.421 của BoW + MLP. Đánh đổi: cần CPU nhiều hơn và encode lúc huấn luyện ≈ 21–29 s, tốn bộ nhớ model 22.7 triệu tham số. LLM không đạt SLA (p95 ≈ 1.2–1.3 s) và nếu dùng cho tất cả thì khoảng 8.000 USD/tháng (zero-shot) hoặc 19.500 USD/tháng (few-shot). Phương án tốt nhất là router: 3c xử lý tin chắc chắn, chỉ tin có p ∈ [0.3, 0.7] gửi LLM (giả định 5% → khoảng 400 USD/tháng, chưa đo).

## 7. AI log và điều đã học

AI log chi tiết nằm ở cuối `day01_lab.ipynb`. Tóm tắt trung thực: Claude Code (AI) đã rà soát phần tôi làm, viết code TODO 3.1–3.3 và 4.1–4.4, soạn 8 dự đoán, câu trả lời Phần 2–5, thử thách cá nhân và lời giải tính tay; tôi chạy lại và đối chiếu số liệu. Riêng thử thách cá nhân và dự đoán đề yêu cầu tự làm, cần được tôi làm lại/giải thích được khi vấn đáp.

Điều hiểu ra hôm nay: loss train thấp không đủ để kết luận mô hình tốt; điểm cao trên tập test cùng phân phối rất lạc quan, chỉ khi đo trên các case khó mới thấy BoW yếu đến mức nào; và LLM mạnh nhưng chậm, đắt nên hợp làm lớp phụ trong thiết kế lai hơn là lớp chính.
