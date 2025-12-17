# app.py
from flask import Flask, render_template_string, Response, jsonify, request, url_for
from ultralytics import YOLO
import cv2, threading, time, json, random, os, numpy as np
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from datetime import datetime

# -------------------------
# Flask App
# -------------------------
# Install YOLOv8


app = Flask(__name__)
MODEL_NAME = "yolov8n.pt"

# -------------------------
# Load YOLO model
# -------------------------
print("Loading YOLOv8 model...")
model = YOLO(MODEL_NAME)
print("Model loaded.")

# -------------------------
# Object database (500+ items)
# -------------------------
DB_FILE = "object_prices.json"
DEFAULT_PRICES = {
    # --- ADD & CORRECT THESE ITEMS ---
    "Cell Phone": 15000, # This name must exactly match the model's output
    "Remote": 200,       # Change from "Remote Control"
    "Tie": 400,          # Add this
    "Book": 500,         # Change from "Textbook"
    "Cat": 0,            # Add this (not a sale item)
    "Tv": 25000,         # Change from "Television"
    "Baseball Bat": 800, # Add this
    "Hair Drier": 1000,  # Change from "Hair Dryer"
    # ---------------------------------
    # Household
    "Chair":500, "Table":1200, "Sofa":5000, "Bed":7000, "Pillow":300,
    "Blanket":800, "Curtain":600, "Carpet":1200, "Mirror":400, "Lamp":700,
    "Fan":1500, "Air Conditioner":25000, "Refrigerator":20000, "Washing Machine":18000,
    "Microwave":7000, "Toaster":1500, "Blender":2500, "Stove":3000, "Oven":10000,
    "Dishwasher":22000, "Cupboard":6000, "Drawer":1500, "Shelf":1000, "Bucket":200,
    "Mop":150, "Broom":100, "Dustpan":80, "Trash Can":300, "Iron":1200,
    "Ironing Board":900, "Hanger":50, "Laundry Basket":400, "Clock":500,
    "Television":25000, "Remote Control":200, "Speaker":1500, "Light Bulb":100,
    "Switch":80, "Plug":50, "Extension Cord":300, "Curtain Rod":250, "Door Mat":150,
    "Side Table":1000, "Bookshelf":2500, "Night Lamp":600, "Vase":300, "Wall Painting":1200,
    # Kitchen
    "Plate":100, "Bowl":80, "Cup":70, "Mug":120, "Glass":90, "Spoon":40,
    "Fork":40, "Knife":150, "Spatula":100, "Tongs":150, "Whisk":120, "Peeler":100,
    "Grater":180, "Rolling Pin":150, "Cutting Board":250, "Pan":800, "Pot":1000,
    "Kettle":600, "Jug":250, "Strainer":150, "Ladle":100, "Tray":200,
    "Measuring Cup":120, "Measuring Spoon":100, "Bottle":100, "Jar":120,
    "Lunch Box":300, "Thermos":400, "Napkin":50, "Apron":150, "Oil Bottle":200,
    "Spice Box":300, "Salt Shaker":80, "Pepper Grinder":150, "Pan Lid":100, "Colander":200,
    "Rice Cooker":2500, "Sandwich Maker":1800, "Coffee Maker":3000, "Tea Pot":500, "Chopping Knife":200,
    "Food Processor":4000, "Ice Tray":50, "Casserole":1200, "Serving Spoon":100, "Soup Bowl":90,
    # Bathroom
    "Toothbrush":50, "Toothpaste":100, "Soap":40, "Shampoo":150, "Conditioner":180,
    "Towel":250, "Hand Towel":120, "Razor":100, "Shaving Cream":120, "Comb":80,
    "Hairbrush":150, "Hair Dryer":1000, "Mirror (Bathroom)":300, "Bucket (Bathroom)":200,
    "Mug (Bathroom)":50, "Bath Mat":250, "Shower Cap":100, "Shower Gel":200, "Loofah":80,
    "Toilet Brush":150, "Soap Dish":100, "Toilet Paper":60, "Towel Rack":200, "Bathroom Shelf":400,
    "Liquid Soap Dispenser":150, "Sponge":50, "Hair Clips":40, "Face Towel":120, "Bathrobe":800,
    # Stationery
    "Pen":10, "Pencil":5, "Eraser":5, "Marker":20, "Highlighter":25,
    "Notebook":50, "Folder":40, "Stapler":150, "Paper":10, "Scissors":120,
    "Glue":50, "Ruler":30, "Calculator":200, "Tape":20, "Printer":8000,
    "Scanner":4000, "Desk":2500, "Chair (Office)":800, "File Cabinet":1500,
    "Lamp (Desk)":400, "Mouse":300, "Keyboard":1000, "Monitor":15000, "Sticky Notes":50,
    "Envelope":10, "Push Pin":5, "Binder Clip":5, "Hole Punch":150, "Index Cards":60,
    "Whiteboard Marker":25, "Whiteboard Eraser":30, "Clipboard":200, "Notebook (Spiral)":50,
    "Pen Stand":100, "Desk Organizer":300, "Paperweight":200, "Stamp Pad":100, "Stamp":50, "Correction Fluid":40,
    # Classroom / Daily Items
    "Textbook":500, "Notebook (School)":50, "Chalk":5, "BlackScientificboard Eraser":30,
    "Board":1000, "Chair (Classroom)":400, "Desk (Classroom)":800, "Marker (Whiteboard)":25,
    "Whiteboard":3000, "Compass":80, "Protractor":60, "Calculator ()":200,
    "Wallet":800, "Watch":2500, "Sunglasses":700, "Keys":100, "Bag":1500,
    "Umbrella":200, "Shoes":1500, "Hat":200, "Water Bottle":80, "Coffee Mug":100,
    "Phone Charger":500, "Headphones (Daily)":700, "Backpack":750, "Lunch Bag":300,
    "Lipstick":150, "Perfume":500, "Notebook (Pocket)":30, "Pocket Mirror":50,
    "Keychain":100, "Hand Sanitizer":120, "Face Mask":50, "Socks":80, "T-Shirt":400,
    "Jeans":1000, "Jacket":2000, "Scarf":300, "Gloves":150, "Earphones":500,
    "Mobile":15000, "Laptop":45000, "Tablet":20000, "Power Bank":1500,
    "Router":3000, "USB Drive":300, "External HDD":5000, "Microphone":800, "Speaker (Bluetooth)":2500,
    "Charger":500, "Cable":100, "Projector":25000, "Camera":15000, "Tripod":1500,
    "Smartwatch":5000, "Drone":15000, "Drone Battery":3000, "Action Camera":12000, "Gaming Console":35000
}

# Create DB file if missing
if not os.path.exists(DB_FILE):
    with open(DB_FILE,"w") as f:
        json.dump(DEFAULT_PRICES,f,indent=4)
        
DB_FILE = "object_prices.json"
with open(DB_FILE,"r") as f:
    object_db = json.load(f)

with open(DB_FILE, "r") as f:
    object_db = json.load(f)

# Remove human-related entries
for key in ["Person", "Man", "Woman", "Boy", "Girl", "People", "Human"]:
    object_db.pop(key, None)


# -------------------------
# Global State
# -------------------------
master_table = []
next_serial = 1
frame_lock = threading.Lock()
annot_lock = threading.Lock()
current_frame = None
annotated_frame = None
class_colors = {}
detection_running = threading.Event()
stop_threads = False
camera_instance = None
camera_thread = None
detection_thread = None

def get_color(name):
    if name not in class_colors:
        class_colors[name] = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
    return class_colors[name]

# -------------------------
# Camera & Detection
# -------------------------
def init_camera():
    global camera_instance
    if camera_instance is None or not camera_instance.isOpened():
        camera_instance = cv2.VideoCapture(0)
        if not camera_instance.isOpened():
            print("Failed to open camera")
            camera_instance = None
            return
        camera_instance.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera_instance.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Camera initialized")

def capture_loop():
    global current_frame, stop_threads
    while not stop_threads:
        if camera_instance is None:
            init_camera()
            time.sleep(0.5)
            continue
        ret, frame = camera_instance.read()
        if ret:
            with frame_lock:
                current_frame = frame.copy()
        else:
            print("Failed to capture frame")
        time.sleep(0.01)

def detection_loop():
    global annotated_frame, master_table, next_serial
    last_annotated = None
    while not stop_threads:
        with frame_lock:
            frame = current_frame.copy() if current_frame is not None else None
        if frame is None:
            time.sleep(0.03)
            continue
        annotated = frame.copy()
        if detection_running.is_set():
            try:
                results = model(frame, conf=0.35, verbose=False)
                current_dets = []
                for res in results:
                    if res.boxes is None: continue
                    boxes = res.boxes
                    for i in range(len(boxes)):

                        x1,y1,x2,y2 = map(int, boxes.xyxy[i].tolist())
                        cls_idx = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        class_name = model.names.get(cls_idx,f"class_{cls_idx}").title()

                                # Skip human detections
                        if class_name.lower() in ["person", "man", "woman", "boy", "girl", "people", "human"]:
                            continue




                    

                        color = get_color(class_name)
                        cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
                        label = f"{class_name} {conf*100:.1f}%"
                        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
                        cv2.rectangle(annotated,(x1,y1-th-10),(x1+tw+6,y1),color,-1)
                        cv2.putText(annotated,label,(x1+3,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
                        current_dets.append({"name": class_name,"confidence": conf})

                # Update master_table
                for det in current_dets:
                    detected_name = det['name']
                    price = float(object_db.get(detected_name, 0.0))
                    found = False
                    for item in master_table:
                        if item['name'] == detected_name:
                            item['confidence'] = max(item['confidence'], det['confidence'])
                            if item['price'] == 0:
                                item['price'] = price
                            found = True
                            break
                    if not found:
                        master_table.append({
                            "serial": next_serial,
                            "name": detected_name,
                            "price": price,
                            "confidence": det['confidence']
                        })
                        next_serial += 1

                last_annotated = annotated.copy()
            except Exception as e:
                print("Detection error:", e)
        else:
            annotated = last_annotated.copy() if last_annotated is not None else frame.copy()
        with annot_lock:
            annotated_frame = annotated.copy()
        time.sleep(0.03)

def start_threads():
    global camera_thread, detection_thread
    init_camera()
    if camera_thread is None:
        camera_thread = threading.Thread(target=capture_loop, daemon=True)
        camera_thread.start()
    if detection_thread is None:
        detection_thread = threading.Thread(target=detection_loop, daemon=True)
        detection_thread.start()

# -------------------------
# Video Generator
# -------------------------
def gen_video():
    global annotated_frame
    while True:
        with annot_lock:
            frame_to_send = annotated_frame.copy() if annotated_frame is not None else np.zeros((480,640,3),dtype=np.uint8)
        ret, buf = cv2.imencode('.jpg', frame_to_send)
        if not ret:
            time.sleep(0.03)
            continue
        jpg = buf.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpg+b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(gen_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------------
# PDF Invoice
# -------------------------
def generate_invoice(table):
    invoices_folder = "invoices"
    if not os.path.exists(invoices_folder):
        os.makedirs(invoices_folder)
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    pdf_path = os.path.join(invoices_folder,f"invoice_{timestamp}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    data = [["#", "Object", "Price (₹)"]]
    total = 0
    for o in table:
        data.append([o["serial"], o["name"], f"{o['price']:.2f}"])
        total += o["price"]
    data.append(["", "Total", f"{total:.2f}"])
    tbl = Table(data, colWidths=[30, 330, 100])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.gray),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('GRID',(0,0),(-1,-1),1,colors.black)
    ]))
    doc.build([tbl])
    return pdf_path

# -------------------------
# HTML Template
# -------------------------


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Object Detection POS</title>
    <style>
        body { font-family: sans-serif; margin:0; display:flex; height:100vh; background:#121212; color:#e0e0e0; }
        #left-panel { width:30%; padding:20px; background:#1e1e1e; display:flex; flex-direction:column; box-sizing:border-box;}
        #right-panel { width:70%; background:#000; display:flex; align-items:center; justify-content:center; border-left:2px solid #333;}
        #video-feed { max-width:100%; max-height:100%; border:2px solid #66b2ff; }
        .header { display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; }
        .header h1 { margin:0; font-size:1.5em; }
        .buttons button { padding:8px 12px; margin-left:5px; border:none; border-radius:4px; cursor:pointer; font-weight:bold; }
        #start-btn { background:#4CAF50; color:white;} 
        #stop-btn { background:#f44336; color:white;} 
        #clear-btn { background:#ff9800; color:white;} 
        #save-btn { background:#2196F3; color:white;} 
        table { width:100%; border-collapse:collapse; margin-top:10px;}
        th, td { padding:8px; text-align:left; border-bottom:1px solid #333;}
        th { background:#333;}
        .actions-cell button { background:none; border:none; color:#66b2ff; cursor:pointer; font-size:1em; margin-right:5px;}
        .total { text-align:right; font-size:1.2em; font-weight:bold; margin-top:10px; color:#4CAF50;}
    </style>
</head>
<body>
    <div id="left-panel">
        <div class="header">
            <h1>POS Object Detection</h1>
            <div class="buttons">
                <button id="start-btn">Start</button>
                <button id="stop-btn">Stop</button>
                <button id="clear-btn">Clear</button>
                <button id="save-btn">Save</button>
            </div>
        </div>

        <div style="margin-bottom:20px;">
            <h2>Manual Add</h2>
            <input type="text" id="manual-name" placeholder="Object Name">
            <input type="number" id="manual-price" placeholder="Price" step="0.01">
            <button id="manual-add-btn">Add</button>
        </div>

        <h2>Detected Objects</h2>
        <table>
            <thead>
                <tr><th>#</th><th>Object</th><th>Price</th><th>Conf</th><th>Actions</th></tr>
            </thead>
            <tbody id="detected-objects-table-body">
                <tr><td colspan="5">No objects detected</td></tr>
            </tbody>
        </table>
        <div class="total" id="total-amount">Total: ₹0.00</div>
    </div>

    <div id="right-panel">
        <img id="video-feed" src="{{ url_for('video_feed') }}" alt="Video Feed">
    </div>

<script>
document.addEventListener('DOMContentLoaded', function(){
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const clearBtn = document.getElementById('clear-btn');
    const saveBtn = document.getElementById('save-btn');
    const manualAddBtn = document.getElementById('manual-add-btn');
    const tableBody = document.getElementById('detected-objects-table-body');
    const totalAmount = document.getElementById('total-amount');

    function fetchTable() {
        fetch('/table')
        .then(res=>res.json())
        .then(data=>updateTable(data))
        .catch(err=>console.error(err));
    }

    function updateTable(data){
        let total = 0;
        tableBody.innerHTML='';
        if(!data || data.length===0){
            tableBody.innerHTML='<tr><td colspan="5">No objects detected</td></tr>';
        } else {
            data.forEach(item=>{
                const row=document.createElement('tr');
                row.innerHTML=`
                    <td>${item.serial}</td>
                    <td>${item.name}</td>
                    <td>${parseFloat(item.price).toFixed(2)}</td>
                    <td>${(item.confidence*100).toFixed(1)}%</td>
                    <td class="actions-cell">
                        <button class="edit-btn" data-serial="${item.serial}">✎</button>
                        <button class="delete-btn" data-serial="${item.serial}">🗑️</button>
                    </td>
                `;
                tableBody.appendChild(row);
                total += parseFloat(item.price||0);
            });
        }
        totalAmount.textContent=`Total: ₹${total.toFixed(2)}`;
    }

    startBtn.onclick = ()=>fetch('/start');
    stopBtn.onclick = ()=>fetch('/stop');
    clearBtn.onclick = ()=>fetch('/clear',{method:'POST'}).then(fetchTable);
    saveBtn.onclick = ()=>fetch('/save',{method:'POST'}).then(res=>res.json()).then(d=>alert('Saved: '+d.file));

    manualAddBtn.onclick = ()=>{
        const name=document.getElementById('manual-name').value.trim();
        const price=parseFloat(document.getElementById('manual-price').value);
        if(!name||isNaN(price)){ alert('Enter name & price'); return;}
        fetch('/manual_add',{method:'POST',headers:{'Content-Type':'application/json'}, body:JSON.stringify({name,price})})
        .then(res=>res.json()).then(r=>{ if(r.success){ fetchTable(); document.getElementById('manual-name').value=''; document.getElementById('manual-price').value=''; }});
    };

    tableBody.onclick=(e)=>{
        const target=e.target;
        if(target.classList.contains('delete-btn')){
            fetch('/delete_item',{method:'POST',headers:{'Content-Type':'application/json'}, body:JSON.stringify({serial:parseInt(target.dataset.serial)})})
            .then(fetchTable);
        }
        if(target.classList.contains('edit-btn')){
            const newName=prompt('Enter new name:');
            if(newName) fetch('/edit_item',{method:'POST',headers:{'Content-Type':'application/json'}, body:JSON.stringify({serial:parseInt(target.dataset.serial), new_name:newName})}).then(fetchTable);
        }
    };

    fetchTable();
    setInterval(fetchTable,1000);
});
</script>
</body>
</html>
"""


# -------------------------
# Flask Routes
# -------------------------
@app.route("/")
def index():
    start_threads()
    return render_template_string(HTML)

@app.route("/start")
def start(): detection_running.set(); return jsonify(success=True)
@app.route("/stop")
def stop(): detection_running.clear(); return jsonify(success=True)
@app.route("/table")
def table(): return jsonify(master_table)
@app.route("/clear",methods=["POST"])
def clear():
    global master_table,next_serial
    master_table.clear()
    next_serial=1
    return jsonify(success=True)
@app.route("/save",methods=["POST"])
def save(): return jsonify(file=generate_invoice(master_table))
@app.route("/manual_add",methods=["POST"])
def manual_add():
    global next_serial
    data=request.json
    name=data.get("name")
    price=float(data.get("price",0))
    if name and price is not None:
        master_table.append({"serial":next_serial,"name":name,"price":price,"confidence":1.0})
        next_serial+=1
        object_db[name]=price
        with open(DB_FILE,"w") as f: json.dump(object_db,f,indent=4)
        return jsonify(success=True)
    return jsonify(success=False)
@app.route("/edit_item",methods=["POST"])
def edit_item():
    data=request.json
    serial=data.get("serial")
    new_name=data.get("new_name")
    for item in master_table:
        if item["serial"]==serial:
            old_name=item["name"]
            item["name"]=new_name
            if old_name in object_db:
                object_db[new_name]=object_db.pop(old_name)
                with open(DB_FILE,"w") as f: json.dump(object_db,f,indent=4)
            return jsonify(success=True)
    return jsonify(success=False)
@app.route("/delete_item",methods=["POST"])
def delete_item():
    global master_table
    data=request.json
    serial=data.get("serial")
    master_table=[i for i in master_table if i["serial"]!=serial]
    return jsonify(success=True)

# -------------------------
# Run Flask
# -------------------------
if __name__ == '__main__':
    start_threads()
    detection_running.set()  # Auto-start detection
    app.run(host='0.0.0.0', port=5000, debug=False)