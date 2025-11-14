import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from collections import deque
import pickle, os
from datetime import datetime
# ==============================
#  CONFIGURATION
# ==============================
SRC1 = "input1.mp4"
SRC2 = "input2.mp4"
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONF = 0.35
SIMILARITY_THRESHOLD = 0.99
MIN_BOX_AREA = 900
# Removed gallery and log file paths as they're not needed
# ==============================
#  YOLO DETECTOR
# ==============================
model = YOLO(YOLO_MODEL_PATH)
# ==============================
#  DEEPSORT TRACKERS
# ==============================
tracker_cam1 = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0,
                        max_cosine_distance=0.5, embedder="mobilenet",
                        half=True, bgr=True, embedder_gpu=True)
tracker_cam2 = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0,
                        max_cosine_distance=0.5, embedder="mobilenet",
                        half=True, bgr=True, embedder_gpu=True)
# ==============================
#  GLOBAL STATE
# ==============================
global_id_counter = 1
global_id_mapping = {}
gallery = {}


# ==============================
#  COLOR PALETTE
# ==============================
palette = [
    (255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),
    (255,165,0),(128,0,128),(255,192,203),(0,128,128),(128,128,0),
    (128,0,0),(0,128,0),(0,0,128),(192,192,192),(255,215,0),
    (210,105,30),(127,255,212),(220,20,60),(75,0,130)
]
def get_color(pid): return palette[pid % len(palette)]

# ==============================
#  REID NETWORK
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet = resnet.to(device).eval()
transform = T.Compose([
    T.ToPILImage(), T.Resize((128,256)), T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@torch.no_grad()
def compute_embedding(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(crop_rgb).unsqueeze(0).to(device)
    emb = resnet(tensor)
    emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-6)
    return emb.squeeze(0).cpu()

def get_gallery_embedding(gid):
    embs = torch.stack(list(gallery[gid]))
    return embs.mean(dim=0)

# ==============================
#  GLOBAL ID MANAGEMENT
# ==============================
def get_or_create_global_id(camera_id, local_id, emb):
    global global_id_counter
    key = (camera_id, local_id)

    if key in global_id_mapping:
        gid = global_id_mapping[key]
        gallery[gid].append(emb)
        return gid

    best_gid, best_sim = None, -1
    for gid, deq in gallery.items():
        if not len(deq): continue
        gallery_emb = get_gallery_embedding(gid)
        sim = torch.dot(gallery_emb, emb).item()
        if sim > best_sim:
            best_sim, best_gid = sim, gid

    if best_sim >= SIMILARITY_THRESHOLD:
        gallery[best_gid].append(emb)
        global_id_mapping[key] = best_gid
        # Match logged in memory only
        return best_gid
    else:
        gid = global_id_counter
        global_id_mapping[key] = gid
        gallery[gid] = deque([emb], maxlen=10)
        global_id_counter += 1
        return gid

# ==============================
#  FRAME PROCESSING
# ==============================
def process_frame(frame, tracker, camera_id):
    h, w = frame.shape[:2]
    results = model(frame, classes=[0], conf=YOLO_CONF, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
            continue
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, None))

    tracks = tracker.update_tracks(detections, frame=frame)
    ids = []
    for track in tracks:
        if not track.is_confirmed(): continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)].copy()
        emb = compute_embedding(crop)
        if emb is None: continue

        gid = get_or_create_global_id(camera_id, track.track_id, emb)
        ids.append(gid)
        color = get_color(gid)
        label = f"ID: {gid}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+5, y1), color, -1)
        cv2.putText(frame, label, (x1+3, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return frame, ids

# ==============================
#  MAIN LOOP
# ==============================
cap1, cap2 = cv2.VideoCapture(SRC1), cv2.VideoCapture(SRC2)
print("Camera A:", cap1.isOpened(), "Camera B:", cap2.isOpened())

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not (ret1 or ret2): break

    ts1 = cap1.get(cv2.CAP_PROP_POS_MSEC)
    ts2 = cap2.get(cv2.CAP_PROP_POS_MSEC)
    if abs(ts1 - ts2) > 40:  # keep roughly in sync
        continue

    if ret1:
        f1, ids1 = process_frame(f1, tracker_cam1, 1)
        cv2.imshow("Camera A", f1)
        print("CamA IDs:", ids1)
    if ret2:
        f2, ids2 = process_frame(f2, tracker_cam2, 2)
        cv2.imshow("Camera B", f2)
        print("CamB IDs:", ids2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release(); cap2.release(); cv2.destroyAllWindows()
