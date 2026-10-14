# Demo đếm lưu lượng giao thông

Ứng dụng dùng YOLOv11 để phát hiện phương tiện và ByteTrack để theo dõi phương tiện qua các khung hình. `main.py` đếm lượt xe đi qua vạch ngang. `main_region.py` khoanh hai polygon làn đường và đếm số xe hiện có trong từng làn.

## Yêu cầu

- Python 3.10 trở lên (khuyến nghị dùng cùng phiên bản Python đã tạo môi trường `.venv`).
- macOS, Linux hoặc Windows có thể cài được các thư viện trong `requirements.txt`.
- CPU vẫn chạy được. CUDA hoặc Apple MPS nếu được PyTorch hỗ trợ có thể tăng tốc; nếu không, Ultralytics sẽ chạy trên CPU.
- Lần chạy đầu cần kết nối mạng nếu video mẫu hoặc trọng số YOLO chưa có trên máy.

## Cài đặt

Tại thư mục dự án, tạo và kích hoạt môi trường ảo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Khi mở terminal mới, cần kích hoạt lại môi trường trước khi chạy:

```bash
source .venv/bin/activate
```

Trên Windows PowerShell, lệnh kích hoạt tương ứng là:

```powershell
.venv\Scripts\Activate.ps1
```

## Chạy demo

Chạy với tuỳ chọn mặc định:

```bash
source .venv/bin/activate
python main.py
```

`main.py` chỉ đếm lượt xe cắt qua vạch ngang (IN/OUT). Nếu cần đếm số xe hiện đang có trong vùng polygon, chạy entry point riêng:

```bash
source .venv/bin/activate
python main_region.py
```

Khi chạy `main_region.py` với giao diện, chương trình dừng ở khung hình đầu để bạn vẽ hai làn trên mặt đường. Click ít nhất ba đỉnh cho làn 1 rồi nhấn Enter; làm tương tự với làn 2. Backspace xoá đỉnh cuối, `R` vẽ lại làn hiện tại, `Q` huỷ. Sau khi chọn xong, chương trình in toạ độ hai vùng ra terminal để bạn có thể dùng lại ở lần sau.

Mặc định chương trình tìm `vietnam_traffic.mp4` trong thư mục hiện tại. Nếu không tìm thấy, chương trình tự tải video mẫu `vehicles.mp4` từ Supervision. Trọng số `yolo11n.pt` cũng được Ultralytics tự tải khi chưa có.

Để tải clip giao thông Phạm Văn Đồng (TP.HCM) mà script `download_clip.py` cấu hình sẵn:

```bash
source .venv/bin/activate
python download_clip.py
python main.py --source vietnam_traffic.mp4
```

Nếu tải clip thất bại, script sẽ chuyển sang video mẫu của Supervision. Có thể dùng video MP4 khác bằng cách chỉ đường dẫn:

```bash
python main.py --source /duong/dan/toi/video.mp4
```

Nhấn `q` trên cửa sổ video để dừng sớm. Trên máy chủ không có giao diện đồ hoạ, thêm `--no-gui`:

```bash
python main.py --source vietnam_traffic.mp4 --no-gui
```

## Tuỳ chọn dòng lệnh

| Tuỳ chọn | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--source` | `vietnam_traffic.mp4` | Đường dẫn video đầu vào. Nếu file không tồn tại, chương trình chuyển sang tải video mẫu `vehicles.mp4`. |
| `--model` | `yolo11n.pt` | Trọng số YOLO. `yolo11n.pt` nhẹ và nhanh nhất trong các lựa chọn được nêu; `yolo11s.pt` cân bằng hơn; `yolo11m.pt` nặng hơn, thường chính xác hơn nhưng chậm hơn. |
| `--imgsz` | `480` | Kích thước ảnh YOLO dùng khi suy luận. Giá trị nhỏ thường nhanh hơn; giá trị lớn hơn có thể nhận diện vật thể nhỏ tốt hơn nhưng tốn thời gian hơn. |
| `--line-y` | `0.65` | (Chỉ `main.py`) Vị trí vạch đếm theo tỷ lệ chiều cao, từ `0.1` đến `0.9`. Ví dụ `0.5` đặt vạch giữa khung hình. |
| `--no-gui` | Tắt | Không mở cửa sổ xem video, phù hợp khi chạy headless. |

Để đặt vùng bằng lệnh hoặc chạy `--no-gui`, cung cấp cả `--lane1-region` và `--lane2-region`. Mỗi đỉnh là cặp `x,y` theo tỷ lệ khung hình: `(0,0)` là góc trên trái, `(1,1)` là góc dưới phải. Cần ít nhất ba đỉnh cho mỗi vùng. Ví dụ:

```bash
python main_region.py \
  --lane1-region "0.05,0.45;0.45,0.45;0.55,0.95;0.05,0.95" \
  --lane2-region "0.55,0.45;0.95,0.45;0.95,0.95;0.55,0.95"
```

Hai vùng làn được tô mờ bằng hai màu khác nhau và có nhãn LANE 1/LANE 2. Hãy vẽ polygon ôm phần mặt đường mà xe đi qua, tránh vùng bảng thống kê ở góc trên trái. Một xe được tính khi điểm đáy giữa bounding box nằm trong vùng làn tương ứng. `NOW` là số xe đang được thấy trong làn ở khung hình hiện tại; `ENTERED` là tổng số ID theo dõi đã xuất hiện trong làn từ đầu video. `NOW` có thể giảm khi xe đi ra; `ENTERED` tăng khi có ID mới đi vào. Nếu bộ theo dõi làm mất ID rồi tạo ID khác cho cùng một xe, `ENTERED` có thể đếm trùng.

Ví dụ: ưu tiên tốc độ hơn nữa trên CPU bằng ảnh suy luận nhỏ hơn:

```bash
python main.py --model yolo11n.pt --imgsz 384
```

Ví dụ: ưu tiên độ chính xác hơn, chấp nhận tốc độ thấp hơn:

```bash
python main.py --model yolo11s.pt --imgsz 640
```

Chương trình tự thu nhỏ video đầu vào nếu chiều rộng vượt 1280 hoặc chiều cao vượt 720, đồng thời giữ nguyên tỷ lệ khung hình. Video nhỏ hơn giới hạn này được giữ nguyên kích thước.

## FPS và tốc độ xử lý

FPS màu xanh trên bảng thống kê là tốc độ xử lý đo được của chương trình, tính xấp xỉ bằng số khung hình xử lý mỗi giây. `main.py` hiển thị lượt IN/OUT; dashboard của `main_region.py` hiển thị `NOW`, `ENTERED` và phân loại xe đang có trong từng làn. Log cũng in FPS sau mỗi 30 khung hình. Con số thực tế phụ thuộc vào CPU/GPU, model, độ phân giải video, `--imgsz` và số phương tiện trong khung hình.

Với video có sẵn, chương trình đọc và xử lý từng khung hình nhanh nhất có thể; nó không chờ để phát theo thời gian thực. FPS video xuất ra được đặt theo FPS của video nguồn, nên nếu tốc độ xử lý thấp hơn FPS nguồn thì thời gian chạy chương trình sẽ dài hơn thời lượng video. Để tăng tốc, dùng `yolo11n.pt`, giảm `--imgsz` (ví dụ 384), hoặc dùng video đầu vào có độ phân giải thấp hơn. Những thay đổi này có thể làm giảm độ chính xác, nhất là với xe nhỏ hoặc ở xa.

## Kết quả

Sau khi chạy, các tệp được ghi vào thư mục `output/`:

- `output/result_demo.mp4`: video do `main.py` tạo, có vạch đếm và lượt IN/OUT.
- `output/region_count.mp4`: video do `main_region.py` tạo, với hai vùng làn, số xe hiện có và tổng xe đã vào từng làn.
- `output/traffic_events.csv`: từng lượt xe cắt qua vạch, gồm số khung hình, thời gian tính từ đầu video, ID theo dõi, mã/tên loại xe và hướng IN/OUT. Tệp chỉ được tạo khi có ít nhất một lượt đếm.

Các lớp phương tiện đang đếm là Bicycle, Car, Motorcycle, Bus và Truck. Vị trí vạch của `main.py` được đặt theo `--line-y`; hướng IN/OUT phụ thuộc vào hướng chuyển động tương đối với vạch và được gán bởi LineZone.

## Khắc phục sự cố

- **Không tìm thấy video:** đặt đúng đường dẫn bằng `--source`; nếu file vẫn không tồn tại, chương trình dùng video mẫu của Supervision.
- **Chạy chậm:** kiểm tra FPS trên dashboard, dùng `--model yolo11n.pt --imgsz 384`, và giảm độ phân giải video đầu vào. Máy không có CUDA/MPS sẽ chạy trên CPU.
- **Không mở được cửa sổ video:** chạy với `--no-gui`.
- **Lỗi thiếu thư viện:** xác nhận đã kích hoạt `.venv` rồi chạy `python -m pip install -r requirements.txt`.
- **Lỗi API Supervision:** môi trường cần dùng các phiên bản đã cài từ `requirements.txt`; trong `.venv` của dự án, lời gọi `LineZoneAnnotator.annotate` tương thích với Supervision 0.30.9.
