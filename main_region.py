"""Count current and cumulative vehicles in two user selected road lanes."""

import argparse

from main import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đếm số phương tiện hiện có trong từng làn đường bằng YOLOv11 + ByteTrack"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="vietnam_traffic.mp4",
        help="Đường dẫn video đầu vào; nếu không tìm thấy sẽ dùng video mẫu",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Weights YOLO, mặc định yolo11n.pt để ưu tiên tốc độ",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=480,
        help="Kích thước ảnh suy luận YOLO; giảm để tăng tốc",
    )
    parser.add_argument(
        "--lane1-region",
        type=str,
        default=None,
        help="Các đỉnh polygon làn 1 dạng x,y chuẩn hóa 0-1, phân cách bằng dấu chấm phẩy",
    )
    parser.add_argument(
        "--lane2-region",
        type=str,
        default=None,
        help="Các đỉnh polygon làn 2 dạng x,y chuẩn hóa 0-1, phân cách bằng dấu chấm phẩy",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Không mở cửa sổ hiển thị video",
    )
    args = parser.parse_args()
    if (args.lane1_region is None) != (args.lane2_region is None):
        parser.error("Cần cung cấp cả --lane1-region và --lane2-region.")
    if args.no_gui and args.lane1_region is None:
        parser.error("Chạy --no-gui cần cung cấp --lane1-region và --lane2-region.")

    region_kwargs = {}
    if args.lane1_region is not None:
        region_kwargs = {
            "lane1_region_spec": args.lane1_region,
            "lane2_region_spec": args.lane2_region,
        }

    run_demo(
        source_path=args.source,
        model_weights=args.model,
        line_y_ratio=0.65,
        show_gui=not args.no_gui,
        imgsz=args.imgsz,
        count_mode="region",
        select_lanes=args.lane1_region is None,
        **region_kwargs,
    )


if __name__ == "__main__":
    main()
