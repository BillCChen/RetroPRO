from fastapi import FastAPI, HTTPException, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
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
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
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


def _build_progress_snapshot(task_id: str, progress: Dict[str, Any]) -> Dict[str, Any]:
    avg_iter_seconds = progress.get("avg_iter_seconds")
    current_iteration = int(progress.get("current_iteration", 0) or 0)
    total_iterations = int(progress.get("total_iterations", 0) or 0)
    remaining_iterations = max(total_iterations - current_iteration, 0)
    eta_seconds = None
    if avg_iter_seconds is not None:
        eta_seconds = round(float(avg_iter_seconds) * remaining_iterations, 2)
    return {
        "task_id": task_id,
        "status": progress.get("status", "queued"),
        "message": progress.get("message", ""),
        "current_iteration": current_iteration,
        "total_iterations": total_iterations,
        "progress_percent": round((current_iteration / total_iterations) * 100, 1) if total_iterations > 0 else 0.0,
        "expanded_nodes": int(progress.get("expanded_nodes", 0) or 0),
        "max_depth": int(progress.get("max_depth", 0) or 0),
        "avg_iter_seconds": avg_iter_seconds,
        "eta_seconds": eta_seconds,
        "result_ready": bool(progress.get("result_ready", False)),
        "error": progress.get("error"),
        "result_url": f"/?task_id={task_id}",
        "updated_at": progress.get("updated_at"),
        "completed_at": progress.get("completed_at"),
    }


def _run_prediction_task(task_id: str, request_data: Dict[str, Any], file_name_prefix: str):
    total_iterations = int(request_data["iterations"])
    iter_elapsed_samples: List[float] = []
    _set_task_progress(
        task_id,
        status="running",
        message="Initializing retrosynthesis planner",
        current_iteration=0,
        total_iterations=total_iterations,
        expanded_nodes=0,
        max_depth=0,
        avg_iter_seconds=None,
        result_ready=False,
        started_at=_iso_now(),
        completed_at=None,
        error=None,
    )

    def progress_callback(progress: Dict[str, Any]):
        iter_elapsed = progress.get("iteration_elapsed_seconds")
        if iter_elapsed is not None and iter_elapsed > 0:
            iter_elapsed_samples.append(float(iter_elapsed))
        avg_iter_seconds = round(sum(iter_elapsed_samples) / len(iter_elapsed_samples), 2) if iter_elapsed_samples else None
        status = progress.get("status", "running")
        message = "Expanding synthesis tree"
        if status == "completed":
            message = "Prediction completed, preparing result view"
        elif status == "finished":
            message = "Search finished, preparing result view"
        _set_task_progress(
            task_id,
            status="running" if status in {"running", "completed", "finished"} else status,
            message=message,
            current_iteration=int(progress.get("current_iteration", 0) or 0),
            total_iterations=int(progress.get("total_iterations", total_iterations) or total_iterations),
            expanded_nodes=int(progress.get("expanded_nodes", 0) or 0),
            max_depth=int(progress.get("max_depth", 0) or 0),
            avg_iter_seconds=avg_iter_seconds,
        )

    try:
        planner = RSPlanner(
            gpu=0,
            use_value_fn=bool(request_data["use_value_fn"]),
            iterations=total_iterations,
            expansion_topk=int(request_data["expansion_topk"]),
            one_step_type=request_data["one_step_type"],
            CCS=bool(request_data["CCS"]),
            radius=request_data.get("radius"),
            primary_css_radius=int(request_data["primary_css_radius"]),
            secondary_css_radius=int(request_data["secondary_css_radius"]),
            starting_mols=STARTING_MOLECULES_CACHE,
            progress_callback=progress_callback,
        )

        result = planner.plan(request_data["smiles"])
        result_data = {
            "task_id": task_id,
            "timestamp": _iso_now(),
            "input_smiles": request_data["smiles"],
            "file_prefix": request_data.get("file_prefix"),
            "parameters": {
                "iterations": request_data["iterations"],
                "expansion_topk": request_data["expansion_topk"],
                "use_value_fn": request_data["use_value_fn"],
                "one_step_type": request_data["one_step_type"],
                "CCS": request_data["CCS"],
                "radius": request_data.get("radius"),
                "primary_css_radius": request_data["primary_css_radius"],
                "secondary_css_radius": request_data["secondary_css_radius"],
            },
            "result": result,
        }

        json_path = os.path.join(RESULT_DIR, f"{file_name_prefix}_{task_id}.json")
        routes_string = None
        if result and isinstance(result, dict):
            routes_string = result.get('routes')
        if routes_string:
            try:
                reaction_list = string2reaction_list(routes_string)
                graph = parse_and_stitch(reaction_list)
                positions = compute_layout(graph, x_step=280, y_step=250, margin=80)
                html_content = build_html(graph, 200, 200, positions, f"Synthesis Path {task_id}")
                html_path = os.path.join(RESULT_DIR, f"{file_name_prefix}_{task_id}_pathview.html")
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                result_data['html_available'] = True
            except Exception as exc:
                print(f"HTML visualization generation failed: {exc}")
                result_data['html_available'] = False
        else:
            result_data['html_available'] = False

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        with open(MAPPING_FILE, 'r+', encoding='utf-8') as f:
            mappings = json.load(f)
            mappings[task_id] = {
                "json": f"{file_name_prefix}_{task_id}.json",
                "html": f"{file_name_prefix}_{task_id}_pathview.html" if result_data.get('html_available') else None
            }
            f.seek(0)
            json.dump(mappings, f, indent=2)
            f.truncate()

        final_iter = result.get("iter") if isinstance(result, dict) and result.get("iter") is not None else _get_task_progress(task_id).get("current_iteration", 0)
        final_status = "completed" if result and result.get("succ") else "failed"
        final_message = "Prediction completed. Redirecting to result page." if final_status == "completed" else "Prediction finished without a valid route."
        _set_task_progress(
            task_id,
            status=final_status,
            message=final_message,
            current_iteration=int(final_iter or 0),
            total_iterations=total_iterations,
            result_ready=True,
            completed_at=_iso_now(),
        )
    except Exception as exc:
        print(f"预测任务失败: {str(exc)}")
        _set_task_progress(
            task_id,
            status="failed",
            message="Prediction failed",
            error=str(exc),
            completed_at=_iso_now(),
            result_ready=False,
        )

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

TASK_PROGRESS: Dict[str, Dict[str, Any]] = {}
TASK_PROGRESS_LOCK = threading.Lock()


def _iso_now() -> str:
    return datetime.now().isoformat()


def _set_task_progress(task_id: str, **fields):
    with TASK_PROGRESS_LOCK:
        current = TASK_PROGRESS.setdefault(task_id, {})
        current.update(fields)
        current["updated_at"] = _iso_now()


def _get_task_progress(task_id: str) -> Dict[str, Any]:
    with TASK_PROGRESS_LOCK:
        return dict(TASK_PROGRESS.get(task_id, {}))


def _load_result_json_by_task_id(task_id: str) -> Dict[str, Any]:
    file_path = get_file_path_by_task_id(task_id, "json")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _canonicalize_smiles(smiles: str) -> str:
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def _safe_parse_iso_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _build_history_record(task_id: str) -> Optional[Dict[str, Any]]:
    try:
        payload = _load_result_json_by_task_id(task_id)
    except Exception:
        return None

    result = payload.get("result") or {}
    timestamp = payload.get("timestamp")
    return {
        "task_id": task_id,
        "timestamp": timestamp,
        "file_prefix": payload.get("file_prefix") or "",
        "input_smiles": payload.get("input_smiles") or "",
        "canonical_smiles": _canonicalize_smiles(payload.get("input_smiles") or ""),
        "succ": bool(result.get("succ")),
        "route_len": result.get("route_len"),
        "iter": result.get("iter"),
        "html_available": bool(payload.get("html_available")),
    }


def _merge_history_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for record in records:
        key = record.get("canonical_smiles") or record.get("input_smiles") or record.get("task_id")
        existing = merged.get(key)
        if existing is None:
            merged[key] = {**record, "merged_count": 1}
            continue
        existing["merged_count"] = int(existing.get("merged_count", 1)) + 1
        if (record.get("timestamp") or "") > (existing.get("timestamp") or ""):
            merged[key] = {
                **record,
                "merged_count": existing["merged_count"],
            }
    return list(merged.values())


def _list_history_records(limit: int = 100, query: str = "", start: str = "", end: str = "") -> List[Dict[str, Any]]:
    if not os.path.exists(MAPPING_FILE):
        return []

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    records = []
    query_lower = query.strip().lower()
    start_dt = _safe_parse_iso_datetime(start)
    end_dt = _safe_parse_iso_datetime(end)

    for task_id in mappings.keys():
        record = _build_history_record(task_id)
        if record is None:
            continue

        record_dt = _safe_parse_iso_datetime(record.get("timestamp", ""))
        if start_dt and (record_dt is None or record_dt < start_dt):
            continue
        if end_dt and (record_dt is None or record_dt > end_dt):
            continue

        if query_lower:
            haystacks = [
                record.get("file_prefix", ""),
                record.get("input_smiles", ""),
                record.get("task_id", ""),
            ]
            if not any(query_lower in (text or "").lower() for text in haystacks):
                continue

        records.append(record)

    records = _merge_history_records(records)
    records.sort(key=lambda item: item.get("timestamp") or "", reverse=True)
    return records[:limit]


def _fingerprint_for_similarity(smiles: str, method: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    method = (method or "ecfp4").lower()
    if method == "scaffold":
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            scaffold = mol
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(scaffold, radius=2, nBits=2048)
    if method == "morgan":
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _compute_similarity(smiles_a: str, smiles_b: str, method: str) -> float:
    fp_a = _fingerprint_for_similarity(smiles_a, method)
    fp_b = _fingerprint_for_similarity(smiles_b, method)
    if fp_a is None or fp_b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _find_similar_records(smiles: str, method: str = "ecfp4", limit: int = 3) -> List[Dict[str, Any]]:
    canonical = _canonicalize_smiles(smiles)
    records = _list_history_records(limit=100, query="", start="", end="")
    scored = []
    for record in records:
        record_smiles = record.get("canonical_smiles") or record.get("input_smiles") or ""
        if not record_smiles or record_smiles == canonical:
            continue
        score = _compute_similarity(canonical, record_smiles, method)
        if score <= 0:
            continue
        scored.append({**record, "similarity": round(score, 4), "method": method})
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored[:limit]


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
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if request.file_prefix:
        file_name_prefix = f"{timestamp}-{request.file_prefix}"
    else:
        file_name_prefix = timestamp
    request_data = request.model_dump()
    request_data["CCS"] = bool(request_data.get("CCS", request_data.get("ccs", False)))
    _set_task_progress(
        task_id,
        status="queued",
        message="Task accepted and queued",
        current_iteration=0,
        total_iterations=int(request.iterations),
        expanded_nodes=0,
        max_depth=0,
        avg_iter_seconds=None,
        result_ready=False,
        created_at=_iso_now(),
        completed_at=None,
        error=None,
    )
    worker = threading.Thread(
        target=_run_prediction_task,
        args=(task_id, request_data, file_name_prefix),
        daemon=True,
    )
    worker.start()
    return PredictionResponse(
        task_id=task_id,
        status="accepted",
        message="预测任务已启动"
    )


@app.get("/api/progress/{task_id}")
async def get_progress(task_id: str):
    progress = _get_task_progress(task_id)
    if not progress:
        raise HTTPException(status_code=404, detail="任务不存在")
    return JSONResponse(_build_progress_snapshot(task_id, progress))


@app.get("/api/history")
async def get_history(
    query: str = Query(default=""),
    start: str = Query(default=""),
    end: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=100),
):
    records = _list_history_records(limit=limit, query=query, start=start, end=end)
    return JSONResponse({"records": records, "count": len(records)})


@app.get("/api/similar")
async def get_similar_examples(
    smiles: str = Query(...),
    method: str = Query(default="ecfp4"),
    limit: int = Query(default=3, ge=1, le=20),
):
    records = _find_similar_records(smiles=smiles, method=method, limit=limit)
    return JSONResponse({"records": records, "count": len(records), "method": method})

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
