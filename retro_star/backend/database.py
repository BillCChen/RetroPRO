"""
数据库模块 - 处理任务参数存储和状态管理
"""

import sqlite3
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

# 数据库文件路径
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'tasks.db')
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'tasks.csv')

class TaskParameters(BaseModel):
    """任务参数模型"""
    # 分子输入
    smiles: str
    
    # 逆合成参数
    model_type: str = "deep_learning"  # 模型选择：template, deep_learning
    use_chemical_semantic_segmentation: bool = True  # 是否启用化学语义分割
    expansion_topk: int = 8  # 单步拓展topk，范围1-10
    max_iterations: int = 100  # 最大迭代次数，范围30-1000
    max_depth: int = 10  # 最大深度，范围2-100
    max_building_blocks: int = 10  # 最大构建块数目，范围1-100
    
    # 化学语义分割参数（仅在启用时有效）
    primary_css_radius: int = 9  # 主要化学语义分割半径，范围1-15
    secondary_css_radius: int = 4  # 次要化学语义分割半径，范围1-15
    
    # 个性化参数
    result_email: str = "2010307209@stu.pku.edu.cn"
    user_name: str = "test"
    user_id: str = "测试案例"
    remarks: str = "无备注"

class TaskStatus(BaseModel):
    """任务状态模型"""
    task_id: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_file: Optional[str] = None

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self.csv_path = CSV_PATH
        self._init_database()
        self._init_csv()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建任务表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                smiles TEXT NOT NULL,
                model_type TEXT DEFAULT 'deep_learning',
                use_chemical_semantic_segmentation INTEGER DEFAULT 1,
                expansion_topk INTEGER DEFAULT 8,
                max_iterations INTEGER DEFAULT 100,
                max_depth INTEGER DEFAULT 10,
                max_building_blocks INTEGER DEFAULT 10,
                primary_css_radius INTEGER DEFAULT 9,
                secondary_css_radius INTEGER DEFAULT 4,
                result_email TEXT DEFAULT '2010307209@stu.pku.edu.cn',
                user_name TEXT DEFAULT 'test',
                user_id TEXT DEFAULT '测试案例',
                remarks TEXT DEFAULT '无备注',
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP NULL,
                completed_at TIMESTAMP NULL,
                error_message TEXT NULL,
                result_file TEXT NULL,
                parameters_json TEXT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_csv(self):
        """初始化CSV文件"""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'task_id', 'smiles', 'model_type', 'use_chemical_semantic_segmentation',
                    'expansion_topk', 'max_iterations', 'max_depth', 'max_building_blocks',
                    'primary_css_radius', 'secondary_css_radius', 'result_email', 'user_name',
                    'user_id', 'remarks', 'status', 'created_at', 'started_at', 
                    'completed_at', 'error_message', 'result_file'
                ])
    
    def create_task(self, params: TaskParameters) -> str:
        """创建新任务"""
        import uuid
        
        task_id = str(uuid.uuid4())
        now = datetime.now()
        
        # 保存到SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO tasks (
                task_id, smiles, model_type, use_chemical_semantic_segmentation,
                expansion_topk, max_iterations, max_depth, max_building_blocks,
                primary_css_radius, secondary_css_radius, result_email, user_name,
                user_id, remarks, status, created_at, parameters_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task_id, params.smiles, params.model_type, 
            int(params.use_chemical_semantic_segmentation),
            params.expansion_topk, params.max_iterations, params.max_depth,
            params.max_building_blocks, params.primary_css_radius, 
            params.secondary_css_radius, params.result_email, params.user_name,
            params.user_id, params.remarks, 'pending', now,
            json.dumps(params.dict())
        ))
        
        conn.commit()
        conn.close()
        
        # 保存到CSV
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                task_id, params.smiles, params.model_type, 
                int(params.use_chemical_semantic_segmentation),
                params.expansion_topk, params.max_iterations, params.max_depth,
                params.max_building_blocks, params.primary_css_radius, 
                params.secondary_css_radius, params.result_email, params.user_name,
                params.user_id, params.remarks, 'pending', now, None, None, None, None
            ])
        
        return task_id
    
    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """获取所有待处理的任务"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM tasks WHERE status = 'pending' ORDER BY created_at ASC
        ''')
        
        columns = [description[0] for description in cursor.description]
        tasks = []
        
        for row in cursor.fetchall():
            task = dict(zip(columns, row))
            # 转换布尔值
            task['use_chemical_semantic_segmentation'] = bool(task['use_chemical_semantic_segmentation'])
            tasks.append(task)
        
        conn.close()
        return tasks
    
    def update_task_status(self, task_id: str, status: str, 
                          error_message: str = None, result_file: str = None):
        """更新任务状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        
        if status == 'running':
            cursor.execute('''
                UPDATE tasks SET status = ?, started_at = ? WHERE task_id = ?
            ''', (status, now, task_id))
        elif status in ['completed', 'failed']:
            cursor.execute('''
                UPDATE tasks SET status = ?, completed_at = ?, 
                error_message = ?, result_file = ? WHERE task_id = ?
            ''', (status, now, error_message, result_file, task_id))
        else:
            cursor.execute('''
                UPDATE tasks SET status = ? WHERE task_id = ?
            ''', (status, task_id))
        
        conn.commit()
        conn.close()
        
        # 同步更新CSV（为了简单，重新写入整个文件）
        self._sync_csv()
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取单个任务详情"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [description[0] for description in cursor.description]
            task = dict(zip(columns, row))
            task['use_chemical_semantic_segmentation'] = bool(task['use_chemical_semantic_segmentation'])
            conn.close()
            return task
        
        conn.close()
        return None
    
    def _sync_csv(self):
        """同步数据库到CSV文件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tasks ORDER BY created_at')
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
        
        conn.close()
    
    def get_statistics(self) -> Dict[str, int]:
        """获取任务统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT status, COUNT(*) as count 
            FROM tasks 
            GROUP BY status
        ''')
        
        stats = {}
        for row in cursor.fetchall():
            stats[row[0]] = row[1]
        
        conn.close()
        return stats

# 全局数据库实例
db = DatabaseManager()