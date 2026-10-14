import os
import time
import argparse
from collections import defaultdict, Counter
import cv2
import numpy as np
import polars as pl
import supervision as sv
from ultralytics import YOLO
from supervision.assets import download_assets, VideoAssets

# Ánh xạ Class ID của COCO sang tên hiển thị và màu sắc (BGR)
CLASS_CONFIG = {
    1: {"name": "Bicycle",    "color": (255, 200, 0)},   # Xanh lơ
    2: {"name": "Car",        "color": (0, 255, 0)},     # Xanh lá
    3: {"name": "Motorcycle", "color": (0, 215, 255)},   # Vàng cam
    5: {"name": "Bus",        "color": (255, 0, 255)},   # Tím
    7: {"name": "Truck",      "color": (0, 69, 255)},    # Đỏ cam
}
TARGET_CLASSES = list(CLASS_CONFIG.keys())


def draw_dashboard(
    frame: np.ndarray,
    counts_in: dict,
    counts_out: dict,
    fps: float,
) -> np.ndarray:
    """Vẽ bảng thống kê bán trong suốt (HUD Overlay) ở góc trái trên màn hình."""
    overlay = frame.copy()
    pad = 20
    box_w, box_h = 370, 260

    # Vẽ nền đen mờ
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (pad, pad), (pad + box_w, pad + box_h), (100, 100, 100), 1)

    # Tiêu đề & FPS
    cv2.putText(frame, "TRAFFIC FLOW COUNTER", (pad + 15, pad + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (pad + 270, pad + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)

    # Header bảng
    y_offset = pad + 65
    cv2.putText(frame, f"{'Vehicle Type':<14} {'IN':>5} {'OUT':>6} {'TOTAL':>7}",
                (pad + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (pad + 15, y_offset + 8), (pad + box_w - 15, y_offset + 8), (100, 100, 100), 1)

    # Dữ liệu từng loại xe
    y_offset += 32
    total_in, total_out = 0, 0
    for cid, cfg in CLASS_CONFIG.items():
        c_in = counts_in[cid]
        c_out = counts_out[cid]
        c_tot = c_in + c_out
        total_in += c_in
        total_out += c_out

        # Chấm màu đại diện cho từng loại xe
        cv2.circle(frame, (pad + 22, y_offset - 5), 5, cfg["color"], -1)
        row_text = f"{cfg['name']:<12} {c_in:>5} {c_out:>6} {c_tot:>7}"
        cv2.putText(frame, row_text, (pad + 35, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        y_offset += 26

    # Dòng tổng cộng (Total)
    cv2.line(frame, (pad + 15, y_offset - 12), (pad + box_w - 15, y_offset - 12), (100, 100, 100), 1)
    tot_text = f"{'TOTAL':<14} {total_in:>5} {total_out:>6} {total_in + total_out:>7}"
    cv2.putText(frame, tot_text, (pad + 15, y_offset + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    return frame


def draw_region_dashboard(
    frame: np.ndarray,
    lane_counts: list[dict],
    lane_entered_counts: list[dict],
    fps: float,
) -> np.ndarray:
    """Draw current occupancy and cumulative arrivals for each lane."""
    pad = 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + 430, pad + 275), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (pad, pad), (pad + 430, pad + 275), (100, 100, 100), 1)
    cv2.putText(frame, "LIVE VEHICLES BY LANE", (pad + 15, pad + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (pad + 330, pad + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)
    lane_colors = [(0, 255, 255), (255, 150, 0)]
    for lane_index, counts in enumerate(lane_counts):
        y = pad + 75 + lane_index * 100
        cv2.circle(frame, (pad + 23, y - 5), 7, lane_colors[lane_index], -1)
        current = sum(counts.values())
        entered = sum(lane_entered_counts[lane_index].values())
        cv2.putText(frame, f"LANE {lane_index + 1}  NOW: {current}  ENTERED: {entered}",
                    (pad + 40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 2, cv2.LINE_AA)
        type_summary_top = "  ".join(
            f"{CLASS_CONFIG[cid]['name']}: {counts.get(cid, 0)}"
            for cid in list(CLASS_CONFIG)[:3]
        )
        type_summary_bottom = "  ".join(
            f"{CLASS_CONFIG[cid]['name']}: {counts.get(cid, 0)}"
            for cid in list(CLASS_CONFIG)[3:]
        )
        cv2.putText(frame, type_summary_top, (pad + 40, y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, type_summary_bottom, (pad + 40, y + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def parse_region(region_spec: str, width: int, height: int) -> np.ndarray:
    """Parse a semicolon-separated list of normalized x,y polygon vertices."""
    try:
        points = [
            tuple(float(value.strip()) for value in pair.split(","))
            for pair in region_spec.split(";")
        ]
        if len(points) < 3 or any(len(point) != 2 for point in points):
            raise ValueError
        if any(not 0 <= coordinate <= 1 for point in points for coordinate in point):
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            "Each lane polygon must contain at least 3 normalized x,y points between 0 and 1, "
            "for example '0.1,0.3;0.9,0.3;0.9,0.9;0.1,0.9'."
        ) from exc

    return np.array(
        [(round(x * width), round(y * height)) for x, y in points],
        dtype=np.int32,
    )


def select_lane_polygons(frame: np.ndarray) -> list[np.ndarray]:
    """Let the user draw the two road lanes on the first video frame."""
    window_name = "Select road lanes"
    lane_colors = [(0, 255, 255), (255, 150, 0)]
    selected = []
    points = []

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    print("[INFO] Vẽ làn 1 rồi làn 2: click ít nhất 3 đỉnh, Enter để chốt; Backspace xoá điểm, R vẽ lại, Q huỷ.")
    try:
        while len(selected) < 2:
            canvas = frame.copy()
            overlay = canvas.copy()
            for lane_index, polygon in enumerate(selected):
                cv2.fillPoly(overlay, [polygon], lane_colors[lane_index])
                cv2.polylines(canvas, [polygon], True, lane_colors[lane_index], 2)
            cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)

            color = lane_colors[len(selected)]
            for point in points:
                cv2.circle(canvas, point, 5, color, -1)
            if len(points) >= 2:
                cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, color, 2)
            cv2.rectangle(canvas, (10, 10), (min(frame.shape[1] - 10, 580), 75), (20, 20, 20), -1)
            cv2.putText(canvas, f"LANE {len(selected) + 1}: click road boundary points",
                        (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, "Enter: confirm  Backspace: undo  R: reset  Q: cancel",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window_name, canvas)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("Đã đóng cửa sổ chọn vùng làn đường.")
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13, 32) and len(points) >= 3:
                selected.append(np.array(points, dtype=np.int32))
                points = []
            elif key in (8, 127) and points:
                points.pop()
            elif key in (ord("r"), ord("R")):
                points = []
            elif key in (ord("q"), ord("Q"), 27):
                raise RuntimeError("Đã huỷ chọn vùng làn đường.")
    finally:
        cv2.destroyWindow(window_name)
    return selected


def run_demo(
    source_path: str,
    model_weights: str,
    line_y_ratio: float,
    show_gui: bool,
    imgsz: int,
    lane1_region_spec: str = "0.02,0.42;0.47,0.42;0.58,0.98;0.02,0.98",
    lane2_region_spec: str = "0.53,0.42;0.98,0.42;0.98,0.98;0.58,0.98",
    count_mode: str = "line",
    select_lanes: bool = False,
):
    os.makedirs("output", exist_ok=True)
    output_filename = "region_count.mp4" if count_mode == "region" else "result_demo.mp4"
    output_video_path = os.path.join("output", output_filename)
    output_csv_path = os.path.join("output", "traffic_events.csv")

    # 1. Nếu chưa có video, tự động tải video mẫu của Supervision
    if not source_path or not os.path.exists(source_path):
        print("[INFO] Không tìm thấy file video chỉ định, tự động tải 'vehicles.mp4'...")
        source_path = download_assets(VideoAssets.VEHICLES)

    # 2. Khởi tạo Model YOLOv11
    print(f"[INFO] Đang nạp mô hình {model_weights}...")
    model = YOLO(model_weights)

    # 3. Giảm kích thước khung hình để tăng tốc xử lý, đặc biệt khi chạy CPU.
    video_info = sv.VideoInfo.from_video_path(source_path)
    source_w, source_h = video_info.width, video_info.height
    scale = min(1.0, 1280 / source_w, 720 / source_h)
    target_w = max(1, int(source_w * scale))
    target_h = max(1, int(source_h * scale))
    if scale < 1.0:
        print(f"[INFO] Thu phóng từ {source_w}x{source_h} về {target_w}x{target_h} để tăng tốc.")
    print(f"[INFO] YOLO inference size: {imgsz}")
    region_polygons = []
    region_zones = []
    if count_mode in ("region", "both"):
        if select_lanes:
            selection_cap = cv2.VideoCapture(source_path)
            has_frame, first_frame = selection_cap.read()
            selection_cap.release()
            if not has_frame:
                raise RuntimeError(f"Không đọc được khung hình đầu tiên từ {source_path} để chọn làn.")
            first_frame = cv2.resize(first_frame, (target_w, target_h))
            region_polygons = select_lane_polygons(first_frame)
            for lane_index, polygon in enumerate(region_polygons):
                spec = ";".join(
                    f"{x / target_w:.3f},{y / target_h:.3f}"
                    for x, y in polygon
                )
                print(f'[INFO] Có thể dùng lại: --lane{lane_index + 1}-region "{spec}"')
        else:
            lane_specs = [lane1_region_spec, lane2_region_spec]
            region_polygons = [parse_region(spec, target_w, target_h) for spec in lane_specs]
        region_zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=[sv.Position.BOTTOM_CENTER],
            )
            for polygon in region_polygons
        ]

    # 4. Thiết lập ByteTrack & Vạch đếm ảo (LineZone)
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=video_info.fps
    )

    line_zone = None
    if count_mode in ("line", "both"):
        line_y = int(target_h * line_y_ratio)
        line_start = sv.Point(int(target_w * 0.05), line_y)
        line_end = sv.Point(int(target_w * 0.95), line_y)
        line_zone = sv.LineZone(
            start=line_start,
            end=line_end,
            triggering_anchors=[sv.Position.BOTTOM_CENTER]
        )

    # 5. Thiết lập Annotators
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(target_w, target_h))
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(target_w, target_h))

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale * 0.85,
        text_thickness=max(1, thickness - 1),
        text_position=sv.Position.TOP_LEFT
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(video_info.fps * 1.2),
        position=sv.Position.BOTTOM_CENTER
    )
    line_annotator = None
    if line_zone is not None:
        line_annotator = sv.LineZoneAnnotator(
            thickness=thickness + 1,
            text_thickness=thickness,
            text_scale=text_scale
        )

    # 6. Biến lưu trữ lịch sử Majority Voting & Bộ đếm
    track_class_history = defaultdict(list)
    counts_in = defaultdict(int)
    counts_out = defaultdict(int)
    lane_entered_counts = [Counter(), Counter()]
    lane_seen_ids = [set(), set()]
    event_logs = []

    cap = cv2.VideoCapture(source_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, video_info.fps, (target_w, target_h))

    frame_idx = 0
    print("[INFO] Bắt đầu xử lý luồng video... (Nhấn 'q' trên cửa sổ video để dừng sớm)")

    while cap.isOpened():
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if (frame.shape[1], frame.shape[0]) != (target_w, target_h):
            frame = cv2.resize(frame, (target_w, target_h))

        # Bước A: Nhận diện phương tiện với YOLOv11
        results = model.predict(
            source=frame,
            classes=TARGET_CLASSES,
            conf=0.25,
            iou=0.5,
            imgsz=imgsz,
            verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        # Bước B: Cập nhật ByteTrack
        detections = byte_tracker.update_with_detections(detections)

        # Bước C: Cập nhật lịch sử Class cho từng Tracker ID (Majority Voting)
        smoothed_classes = []
        for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
            track_class_history[tracker_id].append(int(class_id))
            # Giữ tối đa 30 frames gần nhất để bầu chọn
            if len(track_class_history[tracker_id]) > 30:
                track_class_history[tracker_id].pop(0)
            voted_class = Counter(track_class_history[tracker_id]).most_common(1)[0][0]
            smoothed_classes.append(voted_class)

        if len(smoothed_classes) > 0:
            detections.class_id = np.array(smoothed_classes, dtype=int)

        # Bước D: Kiểm tra phương tiện cắt qua vạch đếm
        lane_counts = [Counter(), Counter()]
        if count_mode in ("region", "both"):
            for lane_index, zone in enumerate(region_zones):
                in_lane = zone.trigger(detections)
                lane_counts[lane_index] = Counter(
                    int(class_id)
                    for class_id, inside in zip(detections.class_id, in_lane)
                    if inside
                )
                for tracker_id, class_id, inside in zip(
                    detections.tracker_id, detections.class_id, in_lane
                ):
                    if inside and tracker_id is not None:
                        tracked_id = int(tracker_id)
                        if tracked_id not in lane_seen_ids[lane_index]:
                            lane_seen_ids[lane_index].add(tracked_id)
                            lane_entered_counts[lane_index][int(class_id)] += 1

        crossed_in, crossed_out = ([], [])
        if line_zone is not None:
            crossed_in, crossed_out = line_zone.trigger(detections)
        timestamp_sec = round(frame_idx / video_info.fps, 2)

        for i, (is_in, is_out) in enumerate(zip(crossed_in, crossed_out)):
            if is_in or is_out:
                t_id = int(detections.tracker_id[i])
                c_id = int(detections.class_id[i])
                direction = "IN" if is_in else "OUT"

                if is_in:
                    counts_in[c_id] += 1
                else:
                    counts_out[c_id] += 1

                event_logs.append({
                    "frame": frame_idx,
                    "time_sec": timestamp_sec,
                    "tracker_id": t_id,
                    "class_id": c_id,
                    "vehicle_type": CLASS_CONFIG[c_id]["name"],
                    "direction": direction
                })

        # Bước E: Vẽ Bounding Box, Quỹ đạo, Vạch đếm và Bảng Dashboard
        labels = [
            f"#{t_id} {CLASS_CONFIG[c_id]['name']}"
            for c_id, t_id in zip(detections.class_id, detections.tracker_id)
        ]

        annotated = frame.copy()
        lane_colors = [(0, 255, 255), (255, 150, 0)]
        overlay = annotated.copy()
        for lane_index, polygon in enumerate(region_polygons):
            cv2.fillPoly(overlay, [polygon], lane_colors[lane_index])
        if region_polygons:
            cv2.addWeighted(overlay, 0.16, annotated, 0.84, 0, annotated)
        for lane_index, polygon in enumerate(region_polygons):
            cv2.polylines(annotated, [polygon], isClosed=True, color=lane_colors[lane_index],
                          thickness=max(2, thickness))
            label_x, label_y = np.mean(polygon, axis=0).astype(int)
            cv2.putText(annotated, f"LANE {lane_index + 1}", (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.6, text_scale),
                        lane_colors[lane_index], max(2, thickness), cv2.LINE_AA)
        annotated = trace_annotator.annotate(scene=annotated, detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        if line_annotator is not None:
            annotated = line_annotator.annotate(frame=annotated, line_counter=line_zone)

        fps = 1.0 / max(time.perf_counter() - t_start, 1e-6)
        if count_mode == "region":
            annotated = draw_region_dashboard(annotated, lane_counts, lane_entered_counts, fps)
        else:
            annotated = draw_dashboard(annotated, counts_in, counts_out, fps)

        writer.write(annotated)

        if show_gui:
            cv2.imshow("Traffic Counter Demo - YOLOv11 + ByteTrack", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Đã dừng bởi người dùng.")
                break

        if frame_idx % 30 == 0:
            print(f" -> Đã xử lý {frame_idx}/{video_info.total_frames} frames ({fps:.1f} FPS)")

    cap.release()
    writer.release()
    if show_gui:
        cv2.destroyAllWindows()

    print(f"\n[SUCCESS] Video demo đã lưu tại: {output_video_path}")

    # 7. Xuất báo cáo tổng hợp bằng Polars
    if count_mode == "region":
        print("[INFO] Tổng lượt xe được theo dõi vào từng làn:")
        for lane_index, counts in enumerate(lane_entered_counts):
            print(f"  Làn {lane_index + 1}: {sum(counts.values())}")
    elif event_logs:
        df = pl.DataFrame(event_logs)
        df.write_csv(output_csv_path)
        print(f"[SUCCESS] Chi tiết từng lượt xe qua vạch đã lưu tại: {output_csv_path}\n")

        summary = (
            df.group_by(["vehicle_type", "direction"])
            .len(name="count")
            .pivot(on="direction", index="vehicle_type", values="count")
            .fill_null(0)
        )
        print("=== BÁO CÁO TỔNG HỢP LƯU LƯỢNG (POLARS) ===")
        print(summary)
    else:
        print("[WARN] Chưa có phương tiện nào cắt qua vạch trong khoảng thời gian chạy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo Đếm Lưu Lượng Giao Thông YOLOv11 + ByteTrack")
    parser.add_argument("--source", type=str, default="vietnam_traffic.mp4",
                        help="Đường dẫn tới file video MP4 hoặc luồng RTSP (nếu bỏ trống sẽ tự tải vehicles.mp4)")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Weights của mô hình: yolo11n.pt (nhanh nhất), yolo11s.pt (cân bằng), yolo11m.pt (chính xác cao)")
    parser.add_argument("--imgsz", type=int, default=480,
                        help="Kích thước ảnh YOLO; giảm để tăng tốc, tăng để cải thiện độ chính xác (mặc định: 480)")
    parser.add_argument(
        "--line-y", type=float, default=0.65,
                        help="Vị trí vạch đếm theo chiều dọc khung hình (từ 0.1 đến 0.9, mặc định 0.65)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Chạy ngầm trên server (Headless mode) không bật cửa sổ cv2.imshow")
    args = parser.parse_args()

    run_demo(
        source_path=args.source,
        model_weights=args.model,
        line_y_ratio=args.line_y,
        show_gui=not args.no_gui,
        imgsz=args.imgsz,
        count_mode="line",
    )
