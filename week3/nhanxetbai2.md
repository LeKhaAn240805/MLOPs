### Hình ví dụ về việc agument được lưu ở aug_samples và két quả results được lưu ở mục results
1. Thiết Lập Dữ Liệu và Kỹ Thuật Data Augmentation
A. Chuẩn bị Dữ liệu
Bộ Dữ liệu Gốc: CIFAR-10.

Dữ liệu sử dụng: 5000 ảnh từ tập huấn luyện.

Lựa chọn Lớp: 5 lớp đã được chọn, mỗi lớp 1000 ảnh (airplane, automobile, bird, cat, deer).

Chia tập: Dữ liệu được chia thành tập huấn luyện (80%) và tập kiểm định (20%) bằng phương pháp train_test_split với stratify.

B. Kỹ thuật Data Augmentation
Các kỹ thuật tăng cường dữ liệu được áp dụng nhằm mở rộng tập huấn luyện và giúp mô hình học các đặc trưng mạnh mẽ hơn, chống overfitting:

Biến đổi Hình học: Lật ngang (RandomFlip), Xoay nhẹ (RandomRotation), Dịch chuyển (RandomTranslation), Phóng to (RandomZoom), Cắt ngẫu nhiên (ZeroPadding2D + RandomCrop).

Biến đổi Màu sắc: Thay đổi độ sáng (RandomBrightness), Thay đổi độ tương phản (RandomContrast).

2. Kiến Trúc và Huấn Luyện Mô Hình
A. Kiến trúc Mô hình (2-Block CNN Tối ưu)
Mô hình được sử dụng là kiến trúc 2-Block Tích chập (VGG-style), được đơn giản hóa từ phiên bản 3-Block để giảm overfitting trên tập dữ liệu nhỏ (5000 ảnh).

Cấu trúc Lõi: Conv(3x3) -> BatchNormalization -> ReLU -> Conv(3x3) -> BatchNormalization -> ReLU -> MaxPooling.

Block 1: 32 filters.

Block 2: 64 filters.

Regularization: L2 Weight Regularization (l2_reg=0.005) được áp dụng cho các lớp Conv và Dense để phạt các trọng số lớn.

Đầu phân loại (Classification Head): GlobalAveragePooling2D (giảm tham số) -> Dense(256) -> Dropout(0.5) -> Dense(5, softmax).

B. Chiến lược Huấn luyện
Mỗi cấu hình (Có Aug và Không Aug) được chạy ít nhất 3 lần (runs=3) và lấy kết quả trung bình.

Callbacks:

EarlyStopping: Theo dõi val_accuracy với patience=10.

ReduceLROnPlateau: Theo dõi val_loss với patience=3, giảm Learning Rate đi một nửa khi loss chững lại, giúp tối ưu hội tụ.

Trường hợp,Độ chính xác Trung bình (Mean Acc),Độ lệch chuẩn (Std Dev),Nhận xét
(1) Dữ liệu Gốc (No Aug),~70.0%,Thấp,"Mức độ chính xác khá tốt, nhưng mô hình dễ bị Overfitting."
(2) Dữ liệu Tăng cường (With Aug),~75.0% - 78.0%,Cao hơn,Độ chính xác cao hơn đáng kể nhờ khả năng tổng quát hóa tốt hơn.

B. Tốc độ Hội tụ và Hiện tượng Overfitting
Biểu đồ Loss (dựa trên các hình minh họa được cung cấp) cho thấy sự khác biệt rõ rệt giữa hai trường hợp:

1. Training Loss (Loss trên Tập Huấn luyện)
Không Augmentation (Đường Đỏ): Loss giảm rất nhanh và đạt mức rất thấp (khoảng 0.6 - 0.7). Điều này chứng tỏ mô hình học thuộc lòng dữ liệu Training rất hiệu quả.

Có Augmentation (Đường Xanh): Loss giảm chậm hơn và ổn định ở mức cao hơn (khoảng 1.0 - 1.2). Điều này là do mô hình luôn phải đối phó với các phiên bản ảnh mới, ngăn chặn việc học thuộc lòng.

2. Validation Loss (Loss trên Tập Kiểm định)
Không Augmentation (Đường Đỏ đứt nét): Loss giảm nhanh trong khoảng 10-20 Epoch đầu, nhưng sau đó dao động mạnh và chững lại (khoảng 1.0 - 1.4). Khoảng cách lớn giữa Training Loss (0.6) và Validation Loss (1.2) là bằng chứng rõ ràng của Overfitting.

Có Augmentation (Đường Xanh đứt nét): Loss giảm ổn định hơn và dừng lại ở mức thấp hơn (khoảng 1.0 - 1.1). Quá trình hội tụ được kiểm soát tốt hơn và mô hình tổng quát hóa tốt hơn với dữ liệu mới.

Việc xây dựng mô hình CNN với kiến trúc được đơn giản hóa (2-Block) kết hợp với L2 Regularization và Adaptive Learning Rate là phương pháp tối ưu cho bộ dữ liệu CIFAR-10 kích thước nhỏ (5000 ảnh).

Data Augmentation không chỉ là yêu cầu bắt buộc mà còn là chìa khóa để cải thiện hiệu năng. Nó giúp tăng độ chính xác cuối cùng lên khoảng 5-8% so với dữ liệu gốc, đồng thời kiểm soát hiệu quả hiện tượng Overfitting.

Các chỉ số Loss và Accuracy trung bình qua 3 lần chạy đã chứng minh tính ổn định và ưu thế của phương pháp tăng cường dữ liệu trong việc xây dựng mô hình phân loại ảnh mạnh mẽ.

