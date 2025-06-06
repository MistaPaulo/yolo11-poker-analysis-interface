"""
Leveraging the existing trained YOLO model for playing-card recognition, this script processes both video and image inputs of a poker table at the end of a round.
It identifies the five community cards and each player's two hole cards, evaluates each hand's ranking, and determines the winner.

Features:
 - Dynamically define ROIs for community cards (dealer) and each player via mouse-driven bounding boxes
 - Run YOLO inference on each frame (video) or a single pass (image), with adjustable input size for higher static-image accuracy
 - Apply non-max suppression within each ROI to keep only the highest-confidence detection per label
 - Persist recent detections up to PERSISTENCE_TIMEOUT seconds to smooth transient misdetections
 - Aggregate detections over a HISTORY_WINDOW to select the most frequent card labels in each ROI
 - Draw fixed blue/red ROIs for table and players, overlaying green bounding boxes with card labels and confidence
 - Display each player's evaluated hand rank below their ROI, and show “Winner” text in the top-left corner
 - “Pause” button (video only) to halt inference and reveal “Export Results” and “Resume”
 - At the end of a video, show buttons for “Export Results,” “Replay,” and “Exit” (export uses a stable snapshot of the last state)
 - On export, write detected community cards and each player's rank/hand to a timestamped .txt file and show “Exported: <filename>” for 2 seconds
 - For image inputs, perform only one inference pass (with IMGSZ_IMAGE) and present “Export Results” and “Exit” buttons after annotation
 - Automatically detect CUDA via PyTorch; if available, move the YOLO model onto the GPU for accelerated inference
"""

import cv2                              # OpenCV for image/video I/O and drawing
import time                             # Time‐related functions (timestamps, sleep)
import argparse                         # Command‐line argument parsing
import numpy as np                      # Numerical operations and array handling
from pathlib import Path                # Filesystem path manipulations
from collections import deque, Counter  # deque for history buffer, Counter for counting labels
from ultralytics import YOLO            # YOLO model for card detection
from treys import Card, Evaluator       # Treys library for poker hand evaluation
from datetime import datetime           # Timestamp formatting for exports
import torch                            # PyTorch for checking CUDA availability and moving model to GPU

# --- Configurations ---
DEFAULT_WEIGHTS     = "weights/poker_best.pt"  # Filepath for the pretrained YOLO model weights used to detect playing cards
CONF_THRESHOLD      = 0.5                      # Minimum confidence threshold for YOLO detections
ANALYSIS_FPS        = 5                        # Number of analysis frames per second (only applies to video input)
IMGSZ               = 1920                     # YOLO input size (pixels) for video frames
IMGSZ_IMAGE         = 1920                     # YOLO input size (pixels) for static images
PERSISTENCE_TIMEOUT = 2.0                      # Time (in seconds) to retain a card detection in memory before discarding it if not re‐detected
HISTORY_WINDOW      = 2.0                      # Duration (in seconds) of the sliding window over which to aggregate detections for smoothing
LABEL_MARGIN_TOP    = 30                       # Vertical pixel margin used to decide whether to draw labels above or below a ROI

BUTTON_W, BUTTON_H = 140, 30
BUTTON_MARGIN      = 10

# Maps YOLO label substrings to Treys format: '10' -> 'T' for rank, and suit letters to lowercase
RANK_MAP = {'10':'T'}
SUIT_MAP = {'C':'c','D':'d','H':'h','S':'s'}

# Global state for ROI and click
drawing     = False
ix = iy     = 0
current_roi = None

mouse_click = False
mouse_x = mouse_y = 0

# Results folder
RESULTS_DIR = Path("poker_analysis_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Handle mouse events for ROI drawing or button clicks
def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, current_roi
    global mouse_click, mouse_x, mouse_y

    mode = param.get('mode', '')
    if mode == 'roi':
        # Allows drawing ROIs on the image/resized display
        bw, bh = param['base_w'], param['base_h']
        win     = param['win']
        _,_,ww,wh = cv2.getWindowImageRect(win)
        scale = min(ww/bw if bw else 1, wh/bh if bh else 1)
        nw, nh = int(bw*scale), int(bh*scale)
        x0, y0 = (ww-nw)//2, (wh-nh)//2
        if not (x0 <= x <= x0+nw and y0 <= y <= y0+nh):
            return
        ix_img = int((x - x0)/scale)
        iy_img = int((y - y0)/scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = ix_img, iy_img
            current_roi = None
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_roi = (
                min(ix, ix_img),
                min(iy, iy_img),
                abs(ix_img - ix),
                abs(iy_img - iy)
            )
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            current_roi = (
                min(ix, ix_img),
                min(iy, iy_img),
                abs(ix_img - ix),
                abs(iy_img - iy)
            )

    elif mode == 'button':
        # Registers button click positions
        if event == cv2.EVENT_LBUTTONUP:
            mouse_click = True
            mouse_x, mouse_y = x, y

# Convert YOLO label (e.g., "10H") to Treys card representation
def to_treys(label):
    r = RANK_MAP.get(label[:-1], label[:-1])
    s = SUIT_MAP.get(label[-1], label[-1].lower())
    return Card.new(r + s)

# Group raw detection results into table cards and player cards by ROI
def group_by_roi(dets, table_roi, player_rois):
    tx,ty,tw,th = table_roi
    table=[]; players={f'player{i+1}':[] for i in range(len(player_rois))}
    for lbl,(x,y,w,h),_ in dets:
        cx,cy = x + w/2, y + h/2
        if tx<=cx<=tx+tw and ty<=cy<=ty+th:
            table.append((lbl,(x,y,w,h)))
        for i,(px,py,pw,ph) in enumerate(player_rois):
            if px<=cx<=px+pw and py<=cy<=py+ph:
                players[f'player{i+1}'].append((lbl,(x,y,w,h)))
    return table, players

# Evaluate poker hands given dealer (board) cards and each player's two cards
def evaluate_hands(assigns):
    # assigns: {'dealer': [ (label,(x,y,w,h)), ... ], 'player1': [...], ...}
    if 'dealer' not in assigns or len(assigns['dealer'])<5:
        return {}
    board = sorted(assigns['dealer'], key=lambda it:it[1][0])[:5]
    seen, board_lbls = set(), []
    for lbl,_ in board:
        if lbl not in seen:
            seen.add(lbl)
            board_lbls.append(lbl)
            if len(board_lbls)==5: break
    board_cards = [to_treys(l) for l in board_lbls]
    ev, results = Evaluator(), {}
    for who,cards in assigns.items():
        if who=='dealer' or len(cards)!=2: continue
        hand = [lbl for lbl,_ in sorted(cards, key=lambda it:it[1][0])]
        try:
            score = ev.evaluate(board_cards, [to_treys(l) for l in hand])
            rank  = ev.class_to_string(ev.get_rank_class(score))
            results[who] = (score, rank, hand)
        except:
            pass
    return results

# Aggregate recent detections into a smoothed result over HISTORY_WINDOW seconds
def aggregate_history(history, player_rois):
    now = time.time()
    while history and now - history[0][0] > HISTORY_WINDOW:
        history.popleft()
    buckets = {}
    for _, assigns in history:
        for who,cards in assigns.items():
            buckets.setdefault(who, []).extend(lbl for lbl,_ in cards)
    combined = {}
    if 'dealer' in buckets:
        top5 = [lbl for lbl,_ in Counter(buckets['dealer']).most_common(5)]
        combined['dealer'] = [(lbl,(0,0,0,0)) for lbl in top5]
    for i in range(len(player_rois)):
        who = f'player{i+1}'
        if who in buckets:
            top2 = [lbl for lbl,_ in Counter(buckets[who]).most_common(2)]
            combined[who] = [(lbl,(0,0,0,0)) for lbl in top2]
    return combined

# Resize and pad a frame to fit into a window while maintaining aspect ratio
def letterbox_canvas(frame, bw, bh, ww, wh):
    if ww<=0 or wh<=0:
        ww, wh = bw, bh
    scale = min(ww/bw if bw else 1, wh/bh if bh else 1)
    nw, nh = max(1,int(bw*scale)), max(1,int(bh*scale))
    resized = cv2.resize(frame, (nw,nh))
    canvas = np.zeros((wh,ww,3), frame.dtype)
    y0, x0 = (wh-nh)//2, (ww-nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

# Draw a clickable button with text on a frame
def draw_button(frame, text, x1, y1):
    x2, y2 = x1+BUTTON_W, y1+BUTTON_H
    cv2.rectangle(frame, (x1,y1),(x2,y2),(50,50,50), -1)
    cv2.putText(frame, text, (x1+5,y1+20),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    return (x1,y1,x2,y2)

# Export detected cards and winner information to a .txt file
def export_to_txt(assigns_dict, results):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = RESULTS_DIR / f"results_{ts}.txt"
    lines = []
    if 'dealer' in assigns_dict:
        cards = assigns_dict['dealer']['cards']
        lines.append("Table cards: " + ", ".join(lbl for lbl,_ in cards))
    for who in sorted(results):
        _, rank, hand = results[who]
        lines.append(f"{who.capitalize()}: {rank} ({', '.join(hand)})")
    if results:
        winner,(_,rank,hand) = min(results.items(), key=lambda kv:kv[1][0])
        lines.append(f"Winner: {winner.capitalize()} - {rank} ({', '.join(hand)})")
    with open(fname, "w") as f:
        f.write("\n".join(lines))
    return fname.name

# Draw fixed ROIs (table and players), all YOLO detections in green, and overlay card labels/ranks
def draw_roi_boxes(frame, table_roi, player_rois, assigns, results, dets_all):
    # 1) First draw every YOLO detection (in green) with its label and confidence
    for lbl,(x,y,w,h),conf in dets_all:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)  # green box
        text = f"{lbl} {conf:.2f}"
        ty = y-10 if y>LABEL_MARGIN_TOP else y+h+20
        cv2.putText(frame, text, (x,ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # 2) Draw the community (dealer) ROI in blue, with dynamic label
    x,y,w,h = table_roi
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)  # blue box for table
    lbls = assigns.get('dealer',{}).get('cards',[])
    txt = f"Table Cards ({', '.join(l for l,_ in lbls)})" if lbls else "Table Cards"
    ty = y+h+20 if y<LABEL_MARGIN_TOP else y-10
    cv2.putText(frame, txt, (x,ty), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

    # 3) Draw each player ROI in red, with that player's evaluated hand rank if available
    for i,(px,py,pw,ph) in enumerate(player_rois, start=1):
        cv2.rectangle(frame,(px,py),(px+pw,py+ph),(0,0,255),2)  # red box for player
        key = f'player{i}'
        if key in results:
            _,rank,hand = results[key]
            t = f"Player {i} - {rank} ({', '.join(hand)})"
        else:
            t = f"Player {i}"
        ty2 = py+ph+20 if py<LABEL_MARGIN_TOP else py-10
        cv2.putText(frame, t, (px,ty2), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

# Overlay the winner text on the frame
def draw_winner(frame, results):
    if not results: return
    w,(_,rank,hand) = min(results.items(), key=lambda kv:kv[1][0])
    txt = f"Winner: {w.capitalize()} - {rank} ({', '.join(hand)})"
    cv2.putText(frame, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

# Check if a point (x,y) lies inside a given rectangle (x1,y1,x2,y2)
def _inside(x, y, rect):
    x1,y1,x2,y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

# Handles video input: defines ROIs, runs YOLO at a reduced frame rate, aggregates detections over time, and provides an interactive display with pause/replay/export.
def process_video(src, model, device):
    """
    Open a video stream (file path or webcam index), let the user draw ROIs for community cards and each player,
    then repeatedly:
      1) Capture frames at normal FPS but run inference only at ANALYSIS_FPS intervals
      2) Perform YOLO detection and retain only the highest-confidence box per label
      3) Group detections into ROIs, persist recent detections (PERSISTENCE_TIMEOUT), and aggregate over HISTORY_WINDOW for stability
      4) Evaluate each player's hand, annotate the frame with:
         - Green boxes + labels/confidence for every detected card
         - Fixed blue box around the community (dealer) ROI
         - Fixed red boxes around each player ROI, with their evaluated rank shown
         - “Winner” text in the top-left when all hands are evaluated
      5) Display an overlay with “Pause” / “Export Results” / “Resume” buttons (or “Export / Replay / Exit” at video end)
      6) On export, write a timestamped .txt file listing community cards, each player's hand + rank, and show a popup message “Exported: <filename>” for 2 seconds
    """
    global mouse_click, mouse_x, mouse_y, current_roi

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[!] Cannot open {src}")
        return

    bw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    bh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)
    skip  = max(int(fps / ANALYSIS_FPS), 1)

    # --- Define ROIs (same as image) ---
    cv2.namedWindow("Define ROIs", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Define ROIs", bw, bh)
    cv2.setMouseCallback("Define ROIs", mouse_callback,
                         {'mode':'roi','win':"Define ROIs",'base_w':bw,'base_h':bh})

    ret, first = cap.read()
    if not ret:
        print("[!] Cannot read first frame")
        return

    rois, labels, idx = [], [], 0
    while True:
        disp = first.copy()
        for r,lbl in zip(rois,labels):
            x,y,w,h = r
            c = (255,0,0) if lbl=="Table Cards" else (0,0,255)
            cv2.rectangle(disp,(x,y),(x+w,y+h),c,2)
            ty = y+h+20 if y<LABEL_MARGIN_TOP else y-10
            cv2.putText(disp, lbl, (x,ty), cv2.FONT_HERSHEY_SIMPLEX,0.6,c,2)
        if current_roi:
            x,y,w,h = current_roi
            cv2.rectangle(disp,(x,y),(x+w,y+h),(0,255,255),2)

        instr = ("Draw Table Cards ROI, ENTER to confirm"
                 if idx==0 else
                 f"Draw Player {idx} ROI, ENTER to confirm, 's' to start")
        cv2.putText(disp, instr, (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        ww,wh = cv2.getWindowImageRect("Define ROIs")[2:]
        canvas = letterbox_canvas(disp, bw, bh, ww, wh)
        cv2.imshow("Define ROIs", canvas)

        k = cv2.waitKey(1) & 0xFF
        if k == 13 and current_roi:  # ENTER
            rois.append(current_roi)
            labels.append("Table Cards" if idx==0 else f"Player {idx}")
            idx += 1
            current_roi = None
            drawing = False
        elif k == ord('s') and idx >= 2:
            break
    cv2.destroyWindow("Define ROIs")

    # --- Main Window ---
    cv2.namedWindow("Poker Analysis", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Poker Analysis", bw, bh)
    cv2.setMouseCallback("Poker Analysis", mouse_callback,
                         {'mode':'button'})

    table_roi   = rois[0]
    player_rois = rois[1:]
    history, smoothed, vis_overlay = deque(), {}, None
    paused = False
    last_export = ""
    export_ts = 0
    frame_idx = 0
    ended = False
    end_smoothed = None
    end_results = None
    last_results = {}
    last_frame = None

    while True:
        if not ended and not paused:
            ret, frame = cap.read()
            if not ret:
                # End of video: final snapshot
                ended = True
                end_smoothed = smoothed.copy()
                end_results  = last_results.copy()
            else:
                last_frame = frame.copy()
                frame_idx += 1
                if frame_idx % skip == 0:
                    # YOLO inference on entire frame, using GPU if available
                    res = model.predict(frame, conf=CONF_THRESHOLD, imgsz=IMGSZ, device=device)[0]
                    raw = [
                        (model.names[int(cls)],
                         (int(x1),int(y1),int(x2-x1),int(y2-y1)),
                         float(score))
                        for x1,y1,x2,y2,score,cls in res.boxes.data.tolist()
                        if score >= CONF_THRESHOLD
                    ]
                    # dedupe by label, keeping highest-confidence
                    best = {}
                    for lbl,bbox,sc in raw:
                        if lbl not in best or sc > best[lbl][2]:
                            best[lbl] = (lbl,bbox,sc)
                    dets = list(best.values())

                    # Group detections by ROI
                    tbl, pls = group_by_roi(dets, table_roi, player_rois)
                    now = time.time()
                    new = {}
                    if len(tbl) >= 5:
                        new['dealer'] = {'cards': sorted(tbl, key=lambda it:it[1][0])[:5], 't': now}
                    for who,cards in pls.items():
                        if len(cards) >= 2:
                            new[who] = {'cards': sorted(cards, key=lambda it:it[1][0])[:2], 't': now}
                    # Persist any ROI that wasn’t redetected but is still within timeout
                    for k,v in smoothed.items():
                        if k not in new and now - v['t'] < PERSISTENCE_TIMEOUT:
                            new[k] = v
                    smoothed = new

                    history.append((now, {k:v['cards'] for k,v in smoothed.items()}))

                    comb = aggregate_history(history, player_rois)
                    results = evaluate_hands(comb)
                    last_results = results

                    vis = frame.copy()
                    # Pass the full list of detections (label, (x,y,w,h), confidence) to draw them in green:
                    draw_roi_boxes(vis, table_roi, player_rois, smoothed, results, dets)
                    draw_winner(vis, results)
                    vis_overlay = vis

        # Display either the most recent overlay or a blank frame if nothing yet
        display = vis_overlay if vis_overlay is not None else (
            last_frame if last_frame is not None else np.zeros((bh,bw,3), np.uint8)
        )
        ww,wh = cv2.getWindowImageRect("Poker Analysis")[2:]
        canvas = letterbox_canvas(display, bw, bh, ww, wh)

        buttons = {}
        x1 = ww - BUTTON_MARGIN - BUTTON_W
        y1 = BUTTON_MARGIN
        if not ended:
            if not paused:
                buttons['pause'] = draw_button(canvas, "Pause", x1, y1)
            else:
                buttons['export_pause'] = draw_button(canvas, "Export results", x1, y1)
                buttons['resume']      = draw_button(canvas, "Resume", x1, y1 + BUTTON_H + BUTTON_MARGIN)
        else:
            buttons['export_end'] = draw_button(canvas, "Export results", x1, y1)
            buttons['replay']     = draw_button(canvas, "Replay", x1, y1 + BUTTON_H + BUTTON_MARGIN)
            buttons['exit']       = draw_button(canvas, "Exit",   x1, y1 + 2*(BUTTON_H + BUTTON_MARGIN))

        if last_export and time.time() - export_ts < 2:
            cv2.putText(canvas, f"Exported: {last_export}",
                        (10, wh - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Poker Analysis", canvas)
        key = cv2.waitKey(delay) & 0xFF

        if mouse_click:
            mx,my = mouse_x, mouse_y
            mouse_click = False
            if not ended:
                if 'pause' in buttons and _inside(mx,my,buttons['pause']):
                    paused = True
                elif 'resume' in buttons and _inside(mx,my,buttons['resume']):
                    paused = False
                elif 'export_pause' in buttons and _inside(mx,my,buttons['export_pause']):
                    last_export = export_to_txt(smoothed, last_results)
                    export_ts = time.time()
            else:
                if 'export_end' in buttons and _inside(mx,my,buttons['export_end']):
                    assigns = end_smoothed if end_smoothed is not None else smoothed
                    results = end_results  if end_results  is not None else last_results
                    last_export = export_to_txt(assigns, results)
                    export_ts = time.time()
                elif 'replay' in buttons and _inside(mx,my,buttons['replay']):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    history.clear(); smoothed.clear()
                    vis_overlay = None; last_frame = None
                    paused = False; ended = False; frame_idx = 0
                    last_export = ""
                    end_smoothed = None; end_results = None
                elif 'exit' in buttons and _inside(mx,my,buttons['exit']):
                    break

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Handles a single‐image input: defines ROIs, runs one pass of YOLO, aggregates detections, and provides an interactive display with export/exit.
def process_image(src, model, device):
    """
    Open a static image from src, allow the user to draw ROIs for community cards and each player, then:
      1) Run YOLO inference once on the full image at IMGSZ_IMAGE resolution
      2) Filter detections to the highest-confidence box per label
      3) Assign detected cards to ROIs, persist them briefly, and evaluate each player's hand
      4) Annotate the image with:
         - Green boxes + labels/confidence for every detected card
         - A fixed blue box around the community (dealer) ROI
         - Fixed red boxes around each player ROI, showing that player's evaluated rank
         - “Winner” text in the top-left if hands are complete
      5) Display the annotated image with “Export Results” and “Exit” buttons
      6) On export, write a timestamped .txt file containing community cards and each player's hand + rank, and show a popup message “Exported: <filename>” for 2 seconds
    """
    global mouse_click, mouse_x, mouse_y, current_roi

    img = cv2.imread(str(src))
    if img is None:
        print(f"[!] Could not open image {src}")
        return

    bw, bh = img.shape[1], img.shape[0]

    # --- Define ROIs (identical to process_video) ---
    cv2.namedWindow("Define ROIs", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Define ROIs", bw, bh)
    cv2.setMouseCallback("Define ROIs", mouse_callback,
                         {'mode':'roi','win':"Define ROIs",'base_w':bw,'base_h':bh})

    rois, labels, idx = [], [], 0
    while True:
        disp = img.copy()
        for r,lbl in zip(rois,labels):
            x,y,w,h = r
            c = (255,0,0) if lbl=="Table Cards" else (0,0,255)
            cv2.rectangle(disp,(x,y),(x+w,y+h),c,2)
            ty = y+h+20 if y<LABEL_MARGIN_TOP else y-10
            cv2.putText(disp, lbl, (x,ty), cv2.FONT_HERSHEY_SIMPLEX,0.6,c,2)
        if current_roi:
            x,y,w,h = current_roi
            cv2.rectangle(disp,(x,y),(x+w,y+h),(0,255,255),2)

        instr = ("Draw Table Cards ROI, ENTER to confirm"
                 if idx==0 else
                 f"Draw Player {idx} ROI, ENTER to confirm, 's' to start")
        cv2.putText(disp, instr, (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        ww,wh = cv2.getWindowImageRect("Define ROIs")[2:]
        canvas = letterbox_canvas(disp, bw, bh, ww, wh)
        cv2.imshow("Define ROIs", canvas)

        k = cv2.waitKey(1) & 0xFF
        if k == 13 and current_roi:  # ENTER
            rois.append(current_roi)
            labels.append("Table Cards" if idx==0 else f"Player {idx}")
            idx += 1
            current_roi = None
            drawing = False
        elif k == ord('s') and idx >= 2:
            break
    cv2.destroyWindow("Define ROIs")

    # --- YOLO inference on static image with larger input size ---
    res = model.predict(img, conf=CONF_THRESHOLD, imgsz=IMGSZ_IMAGE, device=device)[0]
    raw = [
        (model.names[int(cls)],
         (int(x1),int(y1),int(x2-x1),int(y2-y1)),
         float(score))
        for x1,y1,x2,y2,score,cls in res.boxes.data.tolist()
        if score >= CONF_THRESHOLD
    ]
    # dedupe by label, keeping highest-confidence
    best = {}
    for lbl,bbox,sc in raw:
        if lbl not in best or sc > best[lbl][2]:
            best[lbl] = (lbl,bbox,sc)
    dets = list(best.values())

    table_roi   = rois[0]
    player_rois = rois[1:]

    tbl, pls = group_by_roi(dets, table_roi, player_rois)
    now = time.time()
    smoothed = {}
    if len(tbl) >= 5:
        smoothed['dealer'] = {'cards': sorted(tbl, key=lambda it:it[1][0])[:5], 't': now}
    for who,cards in pls.items():
        if len(cards) >= 2:
            smoothed[who] = {'cards': sorted(cards, key=lambda it:it[1][0])[:2], 't': now}

    # For static image, no history; use smoothed directly
    comb = {}
    if 'dealer' in smoothed:
        comb['dealer'] = smoothed['dealer']['cards']
    for who in smoothed:
        if who != 'dealer':
            comb[who] = smoothed[who]['cards']

    results = evaluate_hands(comb)

    # --- Final display and export interface ---
    cv2.namedWindow("Poker Analysis", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Poker Analysis", bw, bh)
    cv2.setMouseCallback("Poker Analysis", mouse_callback,
                         {'mode':'button'})

    last_export = ""
    export_ts = 0

    # Annotated final frame (including all green boxes)
    vis = img.copy()
    draw_roi_boxes(vis, table_roi, player_rois, smoothed, results, dets)
    draw_winner(vis, results)
    vis_overlay = vis

    while True:
        display = vis_overlay
        ww,wh = cv2.getWindowImageRect("Poker Analysis")[2:]
        canvas = letterbox_canvas(display, bw, bh, ww, wh)

        # Buttons: only "Export results" and "Exit"
        buttons = {}
        x1 = ww - BUTTON_MARGIN - BUTTON_W
        y1 = BUTTON_MARGIN
        buttons['export'] = draw_button(canvas, "Export results", x1, y1)
        buttons['exit']   = draw_button(canvas, "Exit",   x1, y1 + BUTTON_H + BUTTON_MARGIN)

        if last_export and time.time() - export_ts < 2:
            cv2.putText(canvas, f"Exported: {last_export}",
                        (10, wh - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Poker Analysis", canvas)
        key = cv2.waitKey(1) & 0xFF

        if mouse_click:
            mx,my = mouse_x, mouse_y
            mouse_click = False
            if 'export' in buttons and _inside(mx,my,buttons['export']):
                last_export = export_to_txt(smoothed, results)
                export_ts = time.time()
            elif 'exit' in buttons and _inside(mx,my,buttons['exit']):
                break

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

# Entry point; parse arguments and decide whether to process video or image
def main():
    p = argparse.ArgumentParser(description="Poker hand analysis")
    p.add_argument("-s","--source", default="0",
                   help="0=webcam/video or file path (image or video)")
    args = p.parse_args()

    weights = Path(DEFAULT_WEIGHTS)
    if not weights.exists():
        print(f"[!] Weights not found: {weights}")
        return

    # Check for CUDA availability
    cuda_available = torch.cuda.is_available()
    print("[INFO] torch.cuda.is_available():", cuda_available)
    if cuda_available:
        print("[INFO] torch.cuda.device_count():", torch.cuda.device_count())
        try:
            print("[INFO] torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
        except Exception:
            pass
    else:
        print("[WARNING] CUDA is not available. YOLO will run on CPU.")

    device = "cuda:0" if cuda_available else "cpu"
    print(f"[INFO] Using device = {device}")

    # Load the YOLO model and move to GPU if available
    model = YOLO(str(weights), task="detect")
    if cuda_available:
        model.to(device)

    src = args.source

    # Detect file extension (if present)
    ext = Path(src).suffix.lower()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    # If user passed only "0" (webcam) or a video extension, call process_video
    if src.isdigit() or ext in {".mp4",".mov",".avi",".mkv"}:
        process_video(int(src) if src.isdigit() else src, model, device)
    # If it’s an image extension, call process_image (using IMGSZ_IMAGE)
    elif ext in image_exts:
        process_image(src, model, device)
    else:
        # If no known extension, try opening as video; if it fails, treat as image
        cap_test = cv2.VideoCapture(src)
        if cap_test.isOpened():
            cap_test.release()
            process_video(src, model, device)
        else:
            cap_test.release()
            process_image(src, model, device)

if __name__=="__main__":
    main()