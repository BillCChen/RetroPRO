"""
RetroTMP 计算工作进程
后台运行，持续检查数据库中的待处理任务并执行计算
"""

import os
import sys
import time
import json
import subprocess
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import DatabaseManager, db

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '..', 'logs', 'worker.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保日志目录存在
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'logs'), exist_ok=True)

class RetroSynthesisWorker:
    """逆合成计算工作进程"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.running = True
        self.run_script_path = os.path.join(os.path.dirname(__file__), '..', 'run.sh')
        self.results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        
        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run(self):
        """主循环 - 持续处理任务"""
        logger.info("RetroTMP 计算工作进程启动")
        logger.info(f"run.sh 路径: {self.run_script_path}")
        logger.info(f"结果目录: {self.results_dir}")
        
        while self.running:
            try:
                # 获取待处理任务
                pending_tasks = self.db.get_pending_tasks()
                
                if pending_tasks:
                    logger.info(f"发现 {len(pending_tasks)} 个待处理任务")
                    
                    for task in pending_tasks:
                        self.process_task(task)
                else:
                    logger.debug("没有待处理任务，等待10秒...")
                
                # 等待一段时间后再次检查
                time.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("收到退出信号，停止工作进程")
                self.running = False
                break
            except Exception as e:
                logger.error(f"工作进程发生错误: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(30)  # 发生错误后等待更长时间
    
    def process_task(self, task: Dict[str, Any]):
        """处理单个任务"""
        task_id = task['task_id']
        logger.info(f"开始处理任务 {task_id}")
        
        try:
            # 更新任务状态为运行中
            self.db.update_task_status(task_id, 'running')
            logger.info(f"任务 {task_id} 状态已更新为 running")
            
            # 准备参数
            params = self._prepare_parameters(task)
            logger.info(f"任务 {task_id} 参数准备完成: {json.dumps(params, ensure_ascii=False)}")
            
            # 生成结果文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"result_{task_id}_{timestamp}.json"
            result_path = os.path.join(self.results_dir, result_filename)
            
            # 调用run.sh执行计算
            success = self._execute_calculation(params, result_path, task_id)
            
            if success:
                # 更新任务状态为完成
                self.db.update_task_status(task_id, 'completed', result_file=result_path)
                logger.info(f"任务 {task_id} 处理完成，结果已保存到 {result_path}")
                
                # 发送结果邮件（如果需要）
                if task.get('result_email'):
                    self._send_result_email(task, result_path)
            else:
                # 更新任务状态为失败
                self.db.update_task_status(task_id, 'failed', error_message="计算过程失败")
                logger.error(f"任务 {task_id} 计算失败")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"处理任务 {task_id} 时发生错误: {error_msg}")
            logger.error(traceback.format_exc())
            
            # 更新任务状态为失败
            self.db.update_task_status(task_id, 'failed', error_message=error_msg)
    
    def _prepare_parameters(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """准备计算参数"""
        params = {
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
            'remarks': task['remarks'],
            'task_id': task['task_id']
        }
        
        return params
    
    def _execute_calculation(self, params: Dict[str, Any], result_path: str, task_id: str) -> bool:
        """执行逆合成计算"""
        try:
            # 检查run.sh是否存在
            if not os.path.exists(self.run_script_path):
                logger.error(f"run.sh 文件不存在: {self.run_script_path}")
                return False
            
            # 确保run.sh有可执行权限
            if not os.access(self.run_script_path, os.X_OK):
                logger.info(f"为 run.sh 添加执行权限")
                os.chmod(self.run_script_path, 0o755)
            
            # 准备命令行参数
            cmd = [
                'bash', self.run_script_path,
                '--smiles', params['smiles'],
                '--model-type', params['model_type'],
                '--expansion-topk', str(params['expansion_topk']),
                '--max-iterations', str(params['max_iterations']),
                '--max-depth', str(params['max_depth']),
                '--max-building-blocks', str(params['max_building_blocks']),
                '--output', result_path,
                '--email', params['result_email']
            ]
            
            # 添加化学语义分割参数
            if params['use_chemical_semantic_segmentation']:
                cmd.extend([
                    '--use-css',
                    '--primary-radius', str(params['primary_css_radius']),
                    '--secondary-radius', str(params['secondary_css_radius'])
                ])
            
            # 记录执行的命令
            cmd_str = ' '.join(cmd)
            logger.info(f"任务 {task_id} 执行命令: {cmd_str}")
            
            # 执行命令
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(self.run_script_path) or '.'
            )
            
            # 等待进程完成
            stdout, stderr = process.communicate()
            
            # 记录输出
            if stdout:
                logger.info(f"任务 {task_id} 标准输出:\n{stdout}")
            if stderr:
                logger.warning(f"任务 {task_id} 标准错误:\n{stderr}")
            
            # 检查返回码
            if process.returncode != 0:
                logger.error(f"任务 {task_id} 执行失败，返回码: {process.returncode}")
                return False
            
            # 检查结果文件是否存在
            if not os.path.exists(result_path):
                logger.error(f"任务 {task_id} 结果文件不存在: {result_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"执行任务 {task_id} 时发生异常: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _send_result_email(self, task: Dict[str, Any], result_path: str):
        """发送结果邮件"""
        try:
            # 这里可以实现邮件发送逻辑
            # 由于您提到run.sh会发送邮件，这里可以留空或实现备用方案
            
            email = task.get('result_email')
            if email:
                logger.info(f"结果邮件将发送到: {email}")
                # 实际邮件发送逻辑...
                
        except Exception as e:
            logger.warning(f"发送结果邮件失败: {str(e)}")

def main():
    """主函数"""
    worker = RetroSynthesisWorker(db)
    
    try:
        worker.run()
    except KeyboardInterrupt:
        print("\n收到退出信号，工作进程已停止")
    except Exception as e:
        logger.error(f"工作进程异常退出: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()