import subprocess
import os
from supervision.assets import download_assets, VideoAssets

def get_vietnam_traffic_clip(output_name="vietnam_traffic.mp4"):
    if os.path.exists(output_name):
        print(f"[INFO] File '{output_name}' đã tồn tại, sẵn sàng chạy demo!")
        return output_name

    url = "http://www.youtube.com/watch?v=VswAfC4fER0"
    print("[INFO] Đang tải 30 giây clip giao thông đường Phạm Văn Đồng (TP.HCM)...")
    
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]",
        "--download-sections", "*00:01:10-00:01:40",
        "-o", output_name,
        url
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] Đã tải thành công: {output_name}")
        return output_name
    except Exception as e:
        print(f"[WARN] Không thể tải từ YouTube ({e}). Chuyển sang tải video mẫu chuẩn của Supervision...")
        return download_assets(VideoAssets.VEHICLES)

if __name__ == "__main__":
    get_vietnam_traffic_clip()
