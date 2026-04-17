"""
RetroTMP 逆合成预测平台 - 后端API
提供任务提交、状态查询、结果下载等功能
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Optional, List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import DatabaseManager, TaskParameters, db

# 创建FastAPI应用
app = FastAPI(
    title="RetroTMP 逆合成预测平台",
    description="基于化学语义分割的智能分子合成路径规划",
    version="2.0.0"
)

# CORS配置 - 允许所有来源（生产环境应限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置路径
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

RUN_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'run.sh')

# 请求模型
class TaskSubmissionRequest(BaseModel):
    """任务提交请求"""
    # 分子输入
    smiles: str
    
    # 逆合成参数
    model_type: str = "deep_learning"  # template, deep_learning
    use_chemical_semantic_segmentation: bool = True
    expansion_topk: int = 8  # 1-10
    max_iterations: int = 100  # 30-1000
    max_depth: int = 10  # 2-100
    max_building_blocks: int = 10  # 1-100
    
    # 化学语义分割参数
    primary_css_radius: int = 9  # 1-15
    secondary_css_radius: int = 4  # 1-15
    
    # 个性化参数
    result_email: str = "2010307209@stu.pku.edu.cn"
    user_name: str = "test"
    user_id: str = "测试案例"
    remarks: str = "无备注"

class TaskSubmissionResponse(BaseModel):
    """任务提交响应"""
    success: bool
    task_id: str
    message: str
    timestamp: str

class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    parameters: Dict[str, Any]

@app.get("/")
async def serve_frontend():
    """服务前端页面"""
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'index.html')
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    else:
        return {"message": "RetroTMP 逆合成预测平台 API 服务运行中"}

@app.post("/api/submit", response_model=TaskSubmissionResponse)
async def submit_task(request: TaskSubmissionRequest):
    """提交逆合成预测任务"""
    try:
        # 验证参数
        if not request.smiles or len(request.smiles) < 5:
            raise HTTPException(status_code=400, detail="无效的SMILES字符串")
        
        if request.expansion_topk < 1 or request.expansion_topk > 10:
            raise HTTPException(status_code=400, detail="expansion_topk必须在1-10之间")
        
        if request.max_iterations < 30 or request.max_iterations > 1000:
            raise HTTPException(status_code=400, detail="max_iterations必须在30-1000之间")
        
        if request.max_depth < 2 or request.max_depth > 100:
            raise HTTPException(status_code=400, detail="max_depth必须在2-100之间")
        
        if request.max_building_blocks < 1 or request.max_building_blocks > 100:
            raise HTTPException(status_code=400, detail="max_building_blocks必须在1-100之间")
        
        if request.use_chemical_semantic_segmentation:
            if request.primary_css_radius < 1 or request.primary_css_radius > 15:
                raise HTTPException(status_code=400, detail="primary_css_radius必须在1-15之间")
            if request.secondary_css_radius < 1 or request.secondary_css_radius > 15:
                raise HTTPException(status_code=400, detail="secondary_css_radius必须在1-15之间")
        
        # 创建任务参数
        params = TaskParameters(
            smiles=request.smiles,
            model_type=request.model_type,
            use_chemical_semantic_segmentation=request.use_chemical_semantic_segmentation,
            expansion_topk=request.expansion_topk,
            max_iterations=request.max_iterations,
            max_depth=request.max_depth,
            max_building_blocks=request.max_building_blocks,
            primary_css_radius=request.primary_css_radius,
            secondary_css_radius=request.secondary_css_radius,
            result_email=request.result_email,
            user_name=request.user_name,
            user_id=request.user_id,
            remarks=request.remarks
        )
        
        # 保存到数据库
        task_id = db.create_task(params)
        
        return TaskSubmissionResponse(
            success=True,
            task_id=task_id,
            message="任务提交成功，正在排队处理",
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"任务提交失败: {str(e)}")

@app.get("/api/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """查询任务状态"""
    task = db.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 加载参数
    if task.get('parameters_json'):
        parameters = json.loads(task['parameters_json'])
    else:
        parameters = {
            'smiles': task['smiles'],
            'model_type': task['model_type'],
            'use_chemical_semantic_segmentation': bool(task['use_chemical_semantic_segmentation']),
            'expansion_topk': task['expansion_topk'],
            'max_iterations': task['max_iterations'],
            'max_depth': task['max_depth'],
            'max_building_blocks': task['max_building_blocks'],
            'primary_css_radius': task['primary_css_radius'],
            'secondary_css_radius': task['secondary_css_radius'],
            'result_email': task['result_email'],
            'user_name': task['user_name'],
            'user_id': task['user_id'],
            'remarks': task['remarks']
        }
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task['status'],
        created_at=task['created_at'],
        started_at=task.get('started_at'),
        completed_at=task.get('completed_at'),
        error_message=task.get('error_message'),
        parameters=parameters
    )

@app.get("/api/download/{task_id}")
async def download_result(task_id: str, file_type: str = "json"):
    """下载任务结果"""
    task = db.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task['status'] != 'completed':
        raise HTTPException(status_code=400, detail="任务尚未完成")
    
    if not task.get('result_file'):
        raise HTTPException(status_code=404, detail="结果文件不存在")
    
    result_path = task['result_file']
    
    if file_type == "json":
        if not result_path.endswith('.json'):
            result_path = result_path.replace('.csv', '.json')
        
        if os.path.exists(result_path):
            return FileResponse(
                result_path,
                media_type="application/json",
                filename=f"retrosynthesis_{task_id}.json"
            )
        else:
            # 如果文件不存在，生成一个包含任务信息的JSON
            result_data = {
                'task_id': task_id,
                'status': task['status'],
                'parameters': json.loads(task['parameters_json']) if task.get('parameters_json') else {},
                'created_at': task['created_at'],
                'completed_at': task.get('completed_at'),
                'note': '原始结果文件可能已被清理'
            }
            
            temp_json = os.path.join(RESULTS_DIR, f"temp_{task_id}.json")
            with open(temp_json, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            return FileResponse(
                temp_json,
                media_type="application/json",
                filename=f"retrosynthesis_{task_id}.json"
            )
    
    raise HTTPException(status_code=400, detail="不支持的文件类型")

@app.get("/api/statistics")
async def get_statistics():
    """获取系统统计信息"""
    stats = db.get_statistics()
    return {
        "statistics": stats,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

# 挂载前端静态文件
frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
