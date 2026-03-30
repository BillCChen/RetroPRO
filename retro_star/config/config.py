"""
RetroTMP 配置文件
包含所有可配置参数
"""

import os
from typing import Dict, Any

# 基础配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 服务器配置
SERVER_CONFIG = {
    "host": "0.0.0.0",  # 监听所有接口，允许校园网访问
    "port": 8000,
    "reload": False,  # 生产环境关闭热重载
    "workers": 1,  # 单工作进程（可根据服务器性能调整）
    "log_level": "info"
}

# 数据库配置
DATABASE_CONFIG = {
    "db_path": os.path.join(BASE_DIR, "database", "tasks.db"),
    "csv_path": os.path.join(BASE_DIR, "database", "tasks.csv")
}

# 计算配置
COMPUTATION_CONFIG = {
    "results_dir": os.path.join(BASE_DIR, "results"),
    "run_script_path": os.path.join(BASE_DIR, "run.sh"),
    "max_concurrent_tasks": 1,  # 同时运行的最大任务数
    "task_timeout": 3600,  # 任务超时时间（秒）
    "retry_attempts": 3,  # 失败重试次数
    "retry_delay": 60  # 重试延迟（秒）
}

# 邮件配置（可选）
EMAIL_CONFIG = {
    "enabled": False,  # 是否启用邮件通知
    "smtp_server": "smtp.pku.edu.cn",  # SMTP服务器
    "smtp_port": 587,
    "username": "",  # 邮箱用户名
    "password": "",  # 邮箱密码（建议使用应用专用密码）
    "from_email": "noreply@retrotmp.pku.edu.cn",
    "use_tls": True
}

# 日志配置
LOGGING_CONFIG = {
    "log_dir": os.path.join(BASE_DIR, "logs"),
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "max_log_size": 50 * 1024 * 1024,  # 50MB
    "backup_count": 5  # 保留的日志文件数
}

# 安全配置
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # CORS允许的源（生产环境应限制）
    "allowed_hosts": ["*"],  # 允许的Host头
    "rate_limit": {
        "enabled": True,
        "requests_per_minute": 60,  # 每分钟最大请求数
        "tasks_per_hour": 10  # 每小时最大任务数
    }
}

# 任务参数默认值
DEFAULT_TASK_PARAMS = {
    "model_type": "deep_learning",
    "use_chemical_semantic_segmentation": True,
    "expansion_topk": 8,
    "max_iterations": 100,
    "max_depth": 10,
    "max_building_blocks": 10,
    "primary_css_radius": 9,
    "secondary_css_radius": 4,
    "result_email": "2010307209@stu.pku.edu.cn",
    "user_name": "test",
    "user_id": "测试案例",
    "remarks": "无备注"
}

# 参数验证规则
PARAMETER_RULES = {
    "smiles": {"type": "string", "min_length": 5, "required": True},
    "model_type": {"type": "string", "allowed": ["template", "deep_learning"], "required": True},
    "use_chemical_semantic_segmentation": {"type": "bool", "required": True},
    "expansion_topk": {"type": "int", "min": 1, "max": 10, "required": True},
    "max_iterations": {"type": "int", "min": 30, "max": 1000, "required": True},
    "max_depth": {"type": "int", "min": 2, "max": 100, "required": True},
    "max_building_blocks": {"type": "int", "min": 1, "max": 100, "required": True},
    "primary_css_radius": {"type": "int", "min": 1, "max": 15, "required": False},
    "secondary_css_radius": {"type": "int", "min": 1, "max": 15, "required": False},
    "result_email": {"type": "email", "required": True},
    "user_name": {"type": "string", "max_length": 100, "required": True},
    "user_id": {"type": "string", "max_length": 100, "required": True},
    "remarks": {"type": "string", "max_length": 500, "required": False}
}

def get_config() -> Dict[str, Any]:
    """获取完整配置"""
    return {
        "server": SERVER_CONFIG,
        "database": DATABASE_CONFIG,
        "computation": COMPUTATION_CONFIG,
        "email": EMAIL_CONFIG,
        "logging": LOGGING_CONFIG,
        "security": SECURITY_CONFIG,
        "defaults": DEFAULT_TASK_PARAMS,
        "rules": PARAMETER_RULES
    }

def load_env_config():
    """从环境变量加载配置"""
    # 服务器配置
    if os.getenv("RETROTMP_HOST"):
        SERVER_CONFIG["host"] = os.getenv("RETROTMP_HOST")
    
    if os.getenv("RETROTMP_PORT"):
        SERVER_CONFIG["port"] = int(os.getenv("RETROTMP_PORT"))
    
    # 邮件配置
    if os.getenv("RETROTMP_EMAIL_ENABLED"):
        EMAIL_CONFIG["enabled"] = os.getenv("RETROTMP_EMAIL_ENABLED").lower() == "true"
    
    if os.getenv("RETROTMP_EMAIL_USER"):
        EMAIL_CONFIG["username"] = os.getenv("RETROTMP_EMAIL_USER")
    
    if os.getenv("RETROTMP_EMAIL_PASS"):
        EMAIL_CONFIG["password"] = os.getenv("RETROTMP_EMAIL_PASS")
    
    # 日志配置
    if os.getenv("RETROTMP_LOG_LEVEL"):
        LOGGING_CONFIG["log_level"] = os.getenv("RETROTMP_LOG_LEVEL").upper()

# 加载环境变量配置
load_env_config()