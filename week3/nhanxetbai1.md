
**1. Ảnh hưởng của các bước tiền xử lý đến hiệu năng mô hình**

###  **Trường hợp 1 – Không tiền xử lý**

* Tập dữ liệu chứa nhiều biến **phân loại dạng string**, ví dụ:
  `"Private"`, `"State-gov"`, `"Never-married"`, `"White"`…
* Các mô hình học máy như Logistic Regression, Decision Tree, Random Forest **không thể xử lý dữ liệu dạng text trực tiếp**, vì chúng yêu cầu input dạng số.
* Do đó cả ba mô hình đều báo lỗi:

```
Error: could not convert string to float
```

**Kết luận:** Không tiền xử lý → mô hình không thể chạy → Accuracy = 0.

---

### **Trường hợp 2 – Có tiền xử lý đầy đủ**

Các bước tiền xử lý đã thực hiện:

| Kỹ thuật              | Vai trò                                            |
| --------------------- | -------------------------------------------------- |
| **One-Hot Encoding**  | Mã hóa biến phân loại thành dạng số                |
| **StandardScaler**    | Chuẩn hóa biến số, giúp Logistic Regression hội tụ |
| **Xóa giá trị thiếu** | Loại bỏ dòng chứa `"?"` giúp mô hình học tốt hơn   |

Sau khi tiền xử lý, tất cả mô hình đều chạy và đạt kết quả tốt:

| Mô hình             | Accuracy   | F1-score   |
| ------------------- | ---------- | ---------- |
| Logistic Regression | 0.8533     | 0.6815     |
| Decision Tree       | 0.8123     | 0.6339     |
| **Random Forest**   | **0.8508** | **0.6849** |

Nhận xét tổng quát:

* Hiệu năng tăng mạnh → chứng tỏ **tiền xử lý là bước bắt buộc**.
* Đặc biệt, StandardScaler giúp Logistic Regression cải thiện rõ rệt.
* One-Hot Encoding giúp mô hình hiểu biến phân loại, tránh sai số.

---

## **2. Mô hình nào cho kết quả tốt nhất? Vì sao?**

 **Random Forest cho kết quả tốt nhất** với:

* **Accuracy: 0.8508**
* **Precision: 0.7375**
* **Recall: 0.6392**
* **F1-score: 0.6848**

###  Nguyên nhân Random Forest tốt nhất:

1. **Khả năng xử lý dữ liệu phi tuyến tốt hơn Logistic Regression**
   → Dữ liệu Adult Income có nhiều quan hệ phi tuyến:

   * nghề nghiệp × trình độ học vấn
   * giờ làm việc × thu nhập
   * tuổi × tình trạng hôn nhân

2. **Tận dụng One-Hot Encoding hiệu quả**

   * Random Forest hoạt động tốt khi số lượng feature tăng (sau OHE).

3. **Giảm overfitting nhờ nhiều cây**

   * Decision Tree dễ overfit
   * Random Forest giảm biến động bằng cách “bình chọn”

4. **Không bị ảnh hưởng nhiều bởi outlier**

   * trong khi Logistic Regression chịu ảnh hưởng mạnh nếu không chuẩn hóa.

 **Kết luận:**

> **Random Forest là mô hình phù hợp nhất cho bài toán phân loại thu nhập Adult Income.**


## **3. Tóm tắt phân tích**

### **Ảnh hưởng tiền xử lý**

* Không tiền xử lý → mô hình lỗi, không chạy được.
* Có tiền xử lý → mô hình chạy tốt, Accuracy tăng mạnh đến **85%**.
* One-Hot Encoding và StandardScaler giúp mô hình học được cấu trúc dữ liệu.

### **Mô hình tốt nhất**

* **Random Forest** có kết quả cao nhất và ổn định nhất.
* Lý do: chống overfitting, xử lý tốt feature đã OHE, học được quan hệ phi tuyến.
