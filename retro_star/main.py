from fastapi import FastAPI, HTTPException, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi import Query
from pydantic import BaseModel
# from retro_star.api import RSPlanner
from api import RSPlanner
import json
import uuid
import os
from datetime import datetime
from pathlib import Path
import hashlib
import secrets
import threading
import time
from rdkit import Chem
from rdkit.Chem import Draw
import networkx as nx
import matplotlib.pyplot as plt
# v2 在 main.py 的顶部添加
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from rdkit.Chem import AllChem
import base64
from io import BytesIO
from collections import deque



# 全局配置
BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = str(BASE_DIR / "results")
os.makedirs(RESULT_DIR, exist_ok=True)
MAPPING_FILE = os.path.join(RESULT_DIR, "file_mappings.json")  # 添加这一行
STATIC_DIR = BASE_DIR / "static"

# 初始化映射文件（如果不存在）
if not os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE, 'w') as f:
        json.dump({}, f)
# 在 main.py 的顶部添加
from typing import Dict, Any, List, Tuple,Optional
from collections import defaultdict
from rdkit.Chem import AllChem
import base64
from io import BytesIO
from collections import deque

# 在 result_data 保存逻辑之前添加这四个函数

def string2reaction_list(reaction_string: str) -> List[str]:
    reactions = reaction_string.split('|')
    def get_reaction(sub_string: str) -> str:
        products, _, reactants = sub_string.split('>')
        return f"{products}>>{reactants}"
    reaction_list = [get_reaction(rxn) for rxn in reactions]
    return reaction_list

def parse_and_stitch(reaction_list: List[str]) -> Dict[str, Any]:
    product_smiles_all, reactant_smiles_all = [], []
    parsed = []
    for rxn in reaction_list:
        prod, rhs = rxn.split(">>")
        prodsmi = prod.strip()
        rhs = rhs.strip()
        reactants = [r.strip() for r in rhs.split(".")] if rhs else []
        product_smiles_all.append(prodsmi)
        reactant_smiles_all.extend(reactants)
        parsed.append((prodsmi, reactants))

    product_set, reactant_set = set(product_smiles_all), set(reactant_smiles_all)
    building_block_smiles = reactant_set - product_set

    nodes = []
    edges = []
    parents = defaultdict(list)
    children = defaultdict(list)

    next_id = 0
    # pending_by_smi: leaves that might be expanded by a later reaction where this smiles is the product
    pending_by_smi: Dict[str, list] = defaultdict(list)
    created_ids_by_smi: Dict[str, list] = defaultdict(list)
    def new_node(smi: str) -> int:
        nonlocal next_id
        nid = next_id
        next_id += 1
        nodes.append({"id": nid, "smi": smi})
        created_ids_by_smi[smi].append(nid)
        return nid

    # Seed: create a node for the first product (root candidate) if needed
    # We'll handle generically in loop: if no pending reactant exists, create a new product node
    for prodsmi, reactants in parsed:
        # find an earlier reactant node with the same smi to expand; if none, create a fresh product node
        if pending_by_smi[prodsmi]:
            prod_id = pending_by_smi[prodsmi].pop(0)
        else:
            prod_id = new_node(prodsmi)

        # add reactant occurrence nodes and connect
        for r in reactants:
            r_id = new_node(r)
            edges.append((prod_id, r_id))
            parents[r_id].append(prod_id)
            children[prod_id].append(r_id)
            # this reactant may be expanded later if it appears as a product in a later reaction
            pending_by_smi[r].append(r_id)

    all_ids = {n["id"] for n in nodes}
    non_roots = set()
    for _, v in edges:
        non_roots.add(v)
    roots = list(all_ids - non_roots)  # nodes that never appear as a child

    building_blocks = set([n["id"] for n in nodes if n["smi"] in building_block_smiles])

    return {
        "nodes": nodes,
        "edges": edges,
        "roots": roots,
        "building_blocks": building_blocks,
    }

def compute_layout(graph: Dict[str, Any], x_step: int = 280, y_step: int = 220, margin: int = 40) -> Dict[int, Tuple[int, int]]:
    nodes = graph["nodes"]
    edges = graph["edges"]
    children = defaultdict(list)
    parents = defaultdict(list)
    for u, v in edges:
        children[u].append(v)
        parents[v].append(u)

    roots = graph["roots"] if graph["roots"] else [nodes[0]["id"]] if nodes else []
    depth = {}
    order_at_depth = defaultdict(list)
    q = deque(roots)
    for r in roots:
        depth[r] = 0
    seen = set(roots)
    while q:
        u = q.popleft()
        d = depth[u]
        order_at_depth[d].append(u)
        for v in children.get(u, []):
            if v not in seen:
                seen.add(v)
                depth[v] = d + 1
                q.append(v)

    for n in nodes:
        if n["id"] not in depth:
            d = 0
            while d in order_at_depth and len(order_at_depth[d]) > 0:
                d += 1
            depth[n["id"]] = d
            order_at_depth[d].append(n["id"])

    positions = {}
    for d in sorted(order_at_depth.keys()):
        items = order_at_depth[d]
        for i, nid in enumerate(items):
            x = margin + d * x_step
            y = margin + i * y_step
            positions[nid] = (x, y)
    return positions

def smiles_to_svg_datauri(smiles: str, size: Tuple[int, int] = (220, 130)) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    
    drawer = Draw.MolDraw2DSVG(size[0], size[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    
    svg = drawer.GetDrawingText()
    b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"

def build_html(graph: Dict[str, Any], card_w: int, card_h: int, positions: Dict[int, Tuple[int, int]], title: str) -> str:
    datauris = {n["id"]: smiles_to_svg_datauri(n["smi"], size=(card_w-20, card_h-50)) for n in graph["nodes"]}
    max_x = max(x for x, y in positions.values()) + card_w + 80
    max_y = max(y for x, y in positions.values()) + card_h + 80

    node_divs = []
    for n in graph["nodes"]:
        nid = n["id"]
        x, y = positions[nid]
        smi = n["smi"]
        is_root = nid in graph["roots"]
        is_block = nid in graph["building_blocks"]
        border = "3px solid #FF3333" if is_block else ("3px solid #FFFFFF" if is_root else "1.5px solid #888")
        shadow = "0 0 24px rgba(255,51,51,0.55)" if is_block else "0 2px 10px rgba(0,0,0,0.4)"
        label = "Building block" if is_block else ("Root" if is_root else "")
        node_divs.append(f"""
  <div class="node" style="left:{x}px; top:{y}px; width:{card_w}px; height:{card_h}px; border:{border}; box-shadow:{shadow}">
    <div class="label">{label}</div>
    <img src="{datauris[nid]}" alt="{smi}"/>
    <div class="smi">{smi}</div>
  </div>
""")

    def center_of(nid): x,y = positions[nid]; return (x + card_w, y + card_h/2)
    def left_of(nid): x,y = positions[nid]; return (x, y + card_h/2)

    edge_paths = []
    for u,v in graph["edges"]:
        x1, y1 = center_of(u); x2, y2 = left_of(v)
        xm = (x1 + x2)/2
        edge_paths.append(f'<path d="M{x1},{y1} C{xm},{y1} {xm},{y2} {x2},{y2}" class="edge" />')
    edges_svg = "\n      ".join(edge_paths)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ background:#0b0b0b; color:#eaeaea; font-family:Inter,system-ui,Arial,sans-serif; margin:0; }}
  h1 {{ font-size:20px; padding:14px 18px; margin:0; background:#0e0e10; border-bottom:1px solid #222; }}
  .canvas {{ position:relative; width:{max_x}px; height:{max_y}px; margin:20px; border:1px solid #222; border-radius:12px;
            background: radial-gradient(1200px 800px at 10% 20%, rgba(0, 229, 255, 0.06), transparent 40%), #0b0b0b; }}
  svg.edges {{ position:absolute; left:0; top:0; width:100%; height:100%; pointer-events:none; }}
  .edge {{ fill:none; stroke:#B0B0B0; stroke-width:2.4; marker-end:url(#arrow); }}
  .node {{ position:absolute; background:rgba(15,15,15,0.92); border-radius:14px; padding:8px; }}
  .node img {{ width:calc(100% - 6px); height:140px; display:block; background:#0d0d0d; border-radius:10px; margin:0 auto 6px; padding:3px; object-fit:contain; }}
  .label {{ font-size:11px; color:#8bdcff; min-height:14px; margin-bottom:4px; }}
  .smi {{ font-family:ui-monospace,Consolas,monospace; font-size:11px; color:#CFCFCF; word-break:break-all; }}
  .legend {{ position:absolute; right:14px; top:14px; font-size:12px; color:#d6f8ff; background:rgba(0,0,0,0.3);
            border:1px solid #1f3b40; border-radius:8px; padding:8px 10px; }}
</style></head>
<body>
<h1>{title}</h1>
<div class="canvas">
  <svg class="edges" viewBox="0 0 {max_x} {max_y}" preserveAspectRatio="none">
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#B0B0B0" />
      </marker>
    </defs>
    {edges_svg}
  </svg>
  {''.join(node_divs)}
  <div class="legend">Cyan glow = building block</div>
</div>
<div class="legend">红色高亮 = 构建块</div>
</body></html>"""
    return html


class PredictionRequest(BaseModel):
    smiles: str
    iterations: int = 100
    expansion_topk: int = 50
    use_value_fn: bool = True
    one_step_type: str   # 可选值: "mlp", "r_smiles"
    CCS: bool = True
    radius: Optional[int] = None
    primary_css_radius: int = 9
    secondary_css_radius: int = 0
    file_prefix: Optional[str] = None  # 添加这一行

class PredictionResponse(BaseModel):
    task_id: str
    status: str
    message: str

# def visualize_path(mol_list, output_path):
#     """生成分子路径图"""
#     mols = [Chem.MolFromSmiles(s) for s in mol_list if s]
#     valid_mols = [mol for mol in mols if mol is not None]
    
#     if valid_mols:
#         img = Draw.MolsToGridImage(
#             valid_mols, 
#             molsPerRow=4,
#             subImgSize=(300, 300),
#             legends=[f"Step {i+1}" for i in range(len(valid_mols))]
#         )
#         img.save(output_path)
# from common.prepare_utils import prepare_starting_molecules
# st_mols = prepare_starting_molecules("dataset/origin_dict.csv")
# 全局缓存变量，在应用启动时加载一次
STARTING_MOLECULES_CACHE = None
from common.prepare_utils import prepare_starting_molecules
from contextlib import asynccontextmanager


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_basic_auth_credentials(auth_header: str) -> Tuple[str, str]:
    """Parse Basic auth header and return (username, password)."""
    if not auth_header:
        return "", ""
    if not auth_header.startswith("Basic "):
        return "", ""
    token = auth_header[len("Basic "):].strip()
    try:
        decoded = base64.b64decode(token).decode("utf-8")
    except Exception:
        return "", ""
    if ":" not in decoded:
        return "", ""
    username, password = decoded.split(":", 1)
    return username, password


def _request_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "").strip()
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _build_auth_guard():
    # Authentication is enabled only when username is set.
    auth_user = os.getenv("RETROTMP_BASIC_AUTH_USER", "").strip()
    auth_password = os.getenv("RETROTMP_BASIC_AUTH_PASSWORD", "")
    auth_password_sha256 = os.getenv("RETROTMP_BASIC_AUTH_PASSWORD_SHA256", "").strip().lower()
    auth_enabled = _env_bool("RETROTMP_BASIC_AUTH_ENABLED", bool(auth_user))

    def is_authorized(request: Request) -> bool:
        if not auth_enabled:
            return True
        if not auth_user:
            return False
        req_user, req_password = _parse_basic_auth_credentials(request.headers.get("authorization", ""))
        if not req_user:
            return False
        if not secrets.compare_digest(req_user, auth_user):
            return False
        if auth_password_sha256:
            req_hash = hashlib.sha256(req_password.encode("utf-8")).hexdigest()
            return secrets.compare_digest(req_hash, auth_password_sha256)
        return secrets.compare_digest(req_password, auth_password)

    return is_authorized, auth_enabled


def _build_predict_rate_limiter():
    enabled = _env_bool("RETROTMP_PREDICT_RATE_LIMIT_ENABLED", True)
    max_requests = int(os.getenv("RETROTMP_PREDICT_RATE_LIMIT_REQUESTS", "6"))
    window_sec = int(os.getenv("RETROTMP_PREDICT_RATE_LIMIT_WINDOW_SEC", "60"))
    bucket: Dict[str, deque] = defaultdict(deque)
    lock = threading.Lock()

    def check(request: Request) -> Tuple[bool, int]:
        if not enabled:
            return True, 0
        now = time.monotonic()
        ip = _request_client_ip(request)
        with lock:
            q = bucket[ip]
            while q and now - q[0] > window_sec:
                q.popleft()
            if len(q) >= max_requests:
                retry_after = max(1, int(window_sec - (now - q[0])))
                return False, retry_after
            q.append(now)
        return True, 0

    return check


def _resolve_starting_molecules_path() -> str:
    """Resolve starting molecules file path with env override."""
    env_path = os.getenv("RETROTMP_STARTING_MOLS_PATH", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
    else:
        candidate = Path(__file__).resolve().parent / "dataset" / "origin_dict.csv"

    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            "Starting molecules file not found. "
            f"Set RETROTMP_STARTING_MOLS_PATH or ensure file exists at: {candidate}"
        )
    return str(candidate)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理（替代 @app.on_event）"""
    """应用启动时预加载起始分子库"""
    global STARTING_MOLECULES_CACHE
    starting_molecules_path = _resolve_starting_molecules_path()
    STARTING_MOLECULES_CACHE = prepare_starting_molecules(starting_molecules_path)
    print(f"✅ 已预加载 {len(STARTING_MOLECULES_CACHE)} 个起始分子")
    yield  # 应用运行中
    
    # ===== 关闭时执行（可选清理）=====
    STARTING_MOLECULES_CACHE = None
    print("✅ 应用关闭：清理缓存")

app = FastAPI(title="RetroTMP 逆合成预测平台",
    lifespan=lifespan  
)

_auth_guard, _auth_enabled = _build_auth_guard()
_predict_rate_limiter = _build_predict_rate_limiter()


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Optional Basic Auth guard for all routes.
    if not _auth_guard(request):
        return PlainTextResponse(
            "Unauthorized",
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": 'Basic realm="RetroPRO"'},
        )
    return await call_next(request)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_route(request: PredictionRequest, raw_request: Request):
    """启动逆合成预测任务"""
    allowed, retry_after = _predict_rate_limiter(raw_request)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many predict requests. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    task_id = str(uuid.uuid4())
    # 生成带前缀的文件名
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if request.file_prefix:
        file_name_prefix = f"{timestamp}-{request.file_prefix}"
    else:
        file_name_prefix = timestamp
    try:

        # 初始化规划器
        planner = RSPlanner(
            gpu=0,  # 可根据实际情况调整
            use_value_fn=request.use_value_fn,
            iterations=int(request.iterations),
            expansion_topk=int(request.expansion_topk),
            one_step_type=request.one_step_type,
            CCS=request.CCS,
            radius=request.radius,
            primary_css_radius=int(request.primary_css_radius),
            secondary_css_radius=int(request.secondary_css_radius),
            starting_mols=STARTING_MOLECULES_CACHE
        )

        
        # 执行预测
        result = planner.plan(request.smiles)
        
        # 保存结果
        result_data = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "input_smiles": request.smiles,
            "file_prefix": request.file_prefix,  # 添加这一行
            "parameters": {
                "iterations": request.iterations,
                "expansion_topk": request.expansion_topk,
                "use_value_fn": request.use_value_fn,
                "one_step_type": request.one_step_type,
                "CCS": request.CCS,
                "radius": request.radius,
                "primary_css_radius": request.primary_css_radius,
                "secondary_css_radius": request.secondary_css_radius,
            },
            "result": result,
            # "route": result.get('route', []) if result else []
        }
        
        # 保存JSON文件
        json_path = os.path.join(RESULT_DIR, f"{file_name_prefix}_{task_id}.json")


                
        # FIX: 生成高级HTML可视化
        # 检查是否有routes字符串
        routes_string = None
        if result and isinstance(result, dict):
            routes_string = result.get('routes')  # 根据实际返回结构调整键名
        if routes_string:
            try:
                # 处理routes字符串并生成HTML
                reaction_list = string2reaction_list(routes_string)
                graph = parse_and_stitch(reaction_list)
                positions = compute_layout(graph, x_step=280, y_step=250, margin=80)
                html_content = build_html(graph, 200, 200, positions, f"Synthesis Path {task_id}")
                
                # 保存HTML文件
                html_path = os.path.join(RESULT_DIR, f"{file_name_prefix}_{task_id}_pathview.html")

                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # 记录HTML路径到result_data
                result_data['html_available'] = True
            except Exception as e:
                print(f"HTML visualization generation failed: {e}")
                result_data['html_available'] = False
        else:
            result_data['html_available'] = False

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        with open(MAPPING_FILE, 'r+') as f:
            mappings = json.load(f)
            mappings[task_id] = {
                "json": f"{file_name_prefix}_{task_id}.json",
                "html": f"{file_name_prefix}_{task_id}_pathview.html" if result_data.get('html_available') else None
            }
            f.seek(0)
            json.dump(mappings, f, indent=2)
        return PredictionResponse(
            task_id=task_id,
            status="success",
            message="预测完成"
        )
        
    except Exception as e:
        print(f"预测任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
def get_file_path_by_task_id(task_id: str, file_type: str) -> str:
    """根据task_id和文件类型获取实际文件路径"""
    if not os.path.exists(MAPPING_FILE):
        raise HTTPException(status_code=404, detail="映射文件不存在")
    
    with open(MAPPING_FILE, 'r') as f:
        mappings = json.load(f)
    
    if task_id not in mappings:
        # 兼容旧版文件（无前缀）
        old_path = os.path.join(RESULT_DIR, f"{task_id}.{file_type}")
        if os.path.exists(old_path):
            return old_path
        raise HTTPException(status_code=404, detail="任务不存在")
    
    filename = mappings[task_id].get(file_type)
    if not filename:
        raise HTTPException(status_code=404, detail="文件类型不存在")
    
    file_path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return file_path

@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """获取预测结果"""
    try:
        file_path = get_file_path_by_task_id(task_id, "json")
        return FileResponse(file_path, media_type="application/json")
    except HTTPException as e:
        raise e

@app.get("/api/download_html/{task_id}")
async def download_html_visualization(task_id: str):
    """下载HTML可视化文件"""
    try:
        file_path = get_file_path_by_task_id(task_id, "html")
        return FileResponse(
            file_path,
            media_type="text/html",
            filename=f"synthesis_path_{task_id}.html"
        )
    except HTTPException as e:
        raise e


@app.get("/api/download/{task_id}/{format_type}")
async def download_result(task_id: str, format_type: str):
    """下载结果文件"""
    try:
        if format_type == "json":
            file_path = get_file_path_by_task_id(task_id, "json")
            media_type = "application/json"
            filename = f"retro_star_{task_id}.json"
        elif format_type == "png":
            # PNG文件通常没有前缀，保持原逻辑或添加映射
            file_path = os.path.join(RESULT_DIR, f"{task_id}.png")
            media_type = "image/png"
            filename = f"retro_star_{task_id}.png"
        else:
            raise HTTPException(status_code=400, detail="不支持的格式")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        return FileResponse(file_path, media_type=media_type, filename=filename)
    except HTTPException as e:
        raise e
@app.post("/api/preview-smiles")
async def preview_smiles(request: dict):
    """生成分子2D预览图"""
    smiles = request.get("smiles", "").strip()
    if not smiles or len(smiles) < 5:
        return {"valid": False, "error": "SMILES太短"}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"valid": False, "error": "无效的SMILES，RDKit无法解析"}
        
        # 生成2D坐标
        Chem.AllChem.Compute2DCoords(mol)
        
        # 生成PNG图像
        img = Draw.MolToImage(mol, size=(400, 250), kekulize=True)
        
        # 转换为base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "valid": True,
            "image": f"data:image/png;base64,{img_base64}"
        }
    except Exception as e:
        return {"valid": False, "error": f"处理出错: {str(e)}"}
# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def serve_frontend():
    """服务前端页面"""
    return FileResponse(str(STATIC_DIR / "index.html"))

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("RETROTMP_HOST", "127.0.0.1")
    port = int(os.getenv("RETROTMP_PORT", "18000"))
    uvicorn.run(app, host=host, port=port)


    # C[C@@H]1Cn2nc(OCc3ccccc3)cc2CN1C(=O)c1ccc(F)cc1
