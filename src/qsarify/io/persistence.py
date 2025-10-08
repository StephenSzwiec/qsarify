"""
SQLite-based persistence system for individual QSAR/QSPR projects.

Each project is a self-contained SQLite file with compressed numpy arrays.
No multi-project management - one file per project like a game save.
"""

import json
import sqlite3
import gzip
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

import numpy as np
import pandas as pd
from dataclasses import asdict, fields

from ..exceptions import QSARifyError

logger = logging.getLogger(__name__)


class PersistenceError(QSARifyError):
    """Base exception for persistence-related errors."""
    pass


def _compress_array(arr: np.ndarray) -> bytes:
    """Compress numpy array to bytes using gzip."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb') as gz_file:
        np.save(gz_file, arr, allow_pickle=False)
    return buffer.getvalue()


def _decompress_array(data: bytes) -> np.ndarray:
    """Decompress bytes to numpy array."""
    buffer = io.BytesIO(data)
    with gzip.GzipFile(fileobj=buffer, mode='rb') as gz_file:
        return np.load(gz_file, allow_pickle=False)


def _serialize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Serialize DataFrame to dict with compressed arrays."""
    return {
        'index': df.index.tolist(),
        'columns': df.columns.tolist(),
        'values_compressed': _compress_array(df.values),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'shape': df.shape
    }


def _deserialize_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """Deserialize DataFrame from dict with compressed arrays."""
    values = _decompress_array(data['values_compressed'])
    df = pd.DataFrame(
        data=values,
        index=data['index'],
        columns=data['columns']
    )
    
    # Restore dtypes
    for col, dtype_str in data['dtypes'].items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype_str)
            except (ValueError, TypeError):
                logger.warning(f"Could not restore dtype {dtype_str} for column {col}")
                
    return df


def _serialize_series(series: pd.Series) -> Dict[str, Any]:
    """Serialize Series to dict with compressed arrays."""
    return {
        'index': series.index.tolist(),
        'values_compressed': _compress_array(series.values),
        'name': series.name,
        'dtype': str(series.dtype),
        'shape': series.shape
    }


def _deserialize_series(data: Dict[str, Any]) -> pd.Series:
    """Deserialize Series from dict with compressed arrays."""
    values = _decompress_array(data['values_compressed'])
    series = pd.Series(
        data=values,
        index=data['index'],
        name=data['name']
    )
    
    # Restore dtype
    try:
        series = series.astype(data['dtype'])
    except (ValueError, TypeError):
        logger.warning(f"Could not restore dtype {data['dtype']} for series {data['name']}")
    
    return series


def _serialize_config(config: Any) -> Dict[str, Any]:
    """Serialize dataclass configuration to JSON-compatible format."""
    if config is None:
        return None
    
    if hasattr(config, '__dataclass_fields__'):
        result = {}
        for field in fields(config):
            value = getattr(config, field.name)
            if isinstance(value, datetime):
                result[field.name] = value.isoformat()
            else:
                result[field.name] = value
        result['__class__'] = config.__class__.__name__
        return result
    else:
        return {'data': str(config), '__class__': config.__class__.__name__}


def create_project_database(db_path: Union[str, Path]) -> None:
    """
    Create SQLite database for a single project.
    
    Args:
        db_path: Path to SQLite database file
    """
    db_path = Path(db_path)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Single project metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS project_metadata (
                project_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                author TEXT,
                created_at TEXT NOT NULL,
                state TEXT NOT NULL,
                target_column TEXT,
                feature_columns TEXT,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Configuration table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS project_configs (
                config_type TEXT PRIMARY KEY,
                config_data TEXT,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Data table for compressed DataFrames and Series
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS project_data (
                data_name TEXT PRIMARY KEY,
                data_type TEXT NOT NULL,
                data_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Model results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS project_models (
                model_name TEXT PRIMARY KEY,
                model_data TEXT NOT NULL,
                statistics_data TEXT,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Validation results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS project_validation (
                validation_type TEXT PRIMARY KEY,
                validation_data TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS project_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                state TEXT NOT NULL,
                details TEXT
            )
        ''')
        
        conn.commit()


def save_project(project, db_path: Union[str, Path]) -> None:
    """
    Save project to its SQLite database file.
    
    Args:
        project: Project instance to save
        db_path: Path to SQLite database file
    """
    from ..project import Project  # Import here to avoid circular imports
    
    db_path = Path(db_path)
    
    # Create database if it doesn't exist
    if not db_path.exists():
        create_project_database(db_path)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Save project metadata
        cursor.execute('''
            INSERT OR REPLACE INTO project_metadata 
            (project_id, name, description, author, created_at, state, 
             target_column, feature_columns, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            project.project_id,
            project.config.name,
            project.config.description,
            project.config.author,
            project.config.created_at.isoformat(),
            project.state.name,
            project.target_column,
            json.dumps(project.feature_columns) if project.feature_columns else None,
            datetime.now().isoformat()
        ))
        
        # Save configurations separately
        configs = {
            'project': _serialize_config(project.config),
            'preprocessing': _serialize_config(project.config.preprocessing_config),
            'optimization': _serialize_config(project.config.optimization_config),
            'genetic': _serialize_config(project.config.genetic_config)
        }
        
        for config_type, config_data in configs.items():
            cursor.execute('''
                INSERT OR REPLACE INTO project_configs (config_type, config_data, updated_at)
                VALUES (?, ?, ?)
            ''', (config_type, json.dumps(config_data), datetime.now().isoformat()))
        
        # Clear and save data
        cursor.execute('DELETE FROM project_data')
        
        # Save DataFrames and Series with compression
        data_items = [
            ('raw_data', 'dataframe', project.raw_data),
            ('X_train', 'dataframe', project.X_train),
            ('X_test', 'dataframe', project.X_test),
            ('y_train', 'series', project.y_train),
            ('y_test', 'series', project.y_test)
        ]
        
        for data_name, data_type, data_obj in data_items:
            if data_obj is not None:
                if data_type == 'dataframe':
                    data_json = json.dumps(_serialize_dataframe(data_obj))
                else:  # series
                    data_json = json.dumps(_serialize_series(data_obj))
                
                cursor.execute('''
                    INSERT INTO project_data (data_name, data_type, data_json, updated_at)
                    VALUES (?, ?, ?, ?)
                ''', (data_name, data_type, data_json, datetime.now().isoformat()))
        
        # Save other data as JSON
        other_data = [
            ('preprocessing_info', 'dict', project.preprocessing_info),
        ]
        
        for data_name, data_type, data_obj in other_data:
            if data_obj is not None:
                cursor.execute('''
                    INSERT INTO project_data (data_name, data_type, data_json, updated_at)
                    VALUES (?, ?, ?, ?)
                ''', (data_name, data_type, json.dumps(data_obj), datetime.now().isoformat()))
        
        # Save model results (JSON serializable parts only)
        cursor.execute('DELETE FROM project_models')
        for model_name, result in project.model_results.items():
            # Filter out non-serializable parts
            serializable_result = {}
            for key, value in result.items():
                if key not in ['model']:  # Skip the actual model object
                    try:
                        json.dumps(value)  # Test if serializable
                        serializable_result[key] = value
                    except (TypeError, ValueError):
                        pass  # Skip non-serializable values
            
            statistics = project.model_statistics.get(model_name, {})
            
            cursor.execute('''
                INSERT INTO project_models (model_name, model_data, statistics_data, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (
                model_name,
                json.dumps(serializable_result),
                json.dumps(statistics),
                datetime.now().isoformat()
            ))
        
        # Save validation results
        cursor.execute('DELETE FROM project_validation')
        for val_type, val_result in project.validation_results.items():
            # Convert validation result to serializable format
            val_data = asdict(val_result) if hasattr(val_result, '__dataclass_fields__') else val_result
            
            cursor.execute('''
                INSERT INTO project_validation (validation_type, validation_data, updated_at)
                VALUES (?, ?, ?)
            ''', (
                val_type,
                json.dumps(val_data),
                datetime.now().isoformat()
            ))
        
        # Save audit log
        cursor.execute('DELETE FROM project_audit_log')
        for log_entry in project.audit_log:
            cursor.execute('''
                INSERT INTO project_audit_log (timestamp, action, state, details)
                VALUES (?, ?, ?, ?)
            ''', (
                log_entry['timestamp'],
                log_entry['action'],
                log_entry['state'],
                json.dumps(log_entry['details'])
            ))
        
        conn.commit()
        
    logger.info(f"Project saved to {db_path}")


def load_project(db_path: Union[str, Path]):
    """
    Load project from its SQLite database file.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        Loaded Project instance
    """
    from ..project import Project, ProjectConfig, ProjectState  # Import here to avoid circular imports
    from ..preprocessing.preprocessing import PreprocessingConfig
    from ..modeling.regressors import OptimizationConfig
    from ..modeling.ga import GeneticConfig
    
    db_path = Path(db_path)
    
    if not db_path.exists():
        raise PersistenceError(f"Project file not found: {db_path}")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Load project metadata
        cursor.execute('''
            SELECT project_id, name, description, author, created_at, state, 
                   target_column, feature_columns
            FROM project_metadata LIMIT 1
        ''')
        
        metadata = cursor.fetchone()
        if not metadata:
            raise PersistenceError("No project metadata found in file")
        
        project_id, name, description, author, created_at_str, state_str, target_column, feature_columns_json = metadata
        
        # Load configurations
        cursor.execute('SELECT config_type, config_data FROM project_configs')
        configs = dict(cursor.fetchall())
        
        # Deserialize configurations
        config_classes = {
            'PreprocessingConfig': PreprocessingConfig,
            'OptimizationConfig': OptimizationConfig,
            'GeneticConfig': GeneticConfig,
            'ProjectConfig': ProjectConfig
        }
        
        def deserialize_config(config_json: str, config_classes: dict) -> Any:
            if not config_json or config_json == 'null':
                return None
            data = json.loads(config_json)
            if data is None:
                return None
            class_name = data.pop('__class__', None)
            if class_name not in config_classes:
                return None
            config_class = config_classes[class_name]
            # Handle datetime fields
            for field in fields(config_class):
                if field.name in data and field.type == datetime:
                    data[field.name] = datetime.fromisoformat(data[field.name])
            return config_class(**data)
        
        preprocessing_config = deserialize_config(configs.get('preprocessing'), config_classes)
        optimization_config = deserialize_config(configs.get('optimization'), config_classes)
        genetic_config = deserialize_config(configs.get('genetic'), config_classes)
        
        # Create project config
        project_config = ProjectConfig(
            name=name,
            description=description,
            author=author,
            created_at=datetime.fromisoformat(created_at_str),
            preprocessing_config=preprocessing_config,
            optimization_config=optimization_config,
            genetic_config=genetic_config,
            auto_checkpoint=False  # Disable during loading
        )
        
        # Create project instance
        project = Project(project_config, db_path)
        project.project_id = project_id
        project.state = ProjectState[state_str]
        project.target_column = target_column
        project.feature_columns = json.loads(feature_columns_json) if feature_columns_json else None
        
        # Load data
        cursor.execute('SELECT data_name, data_type, data_json FROM project_data')
        
        for data_name, data_type, data_json in cursor.fetchall():
            if data_type == 'dataframe':
                data = _deserialize_dataframe(json.loads(data_json))
                setattr(project, data_name, data)
            elif data_type == 'series':
                data = _deserialize_series(json.loads(data_json))
                setattr(project, data_name, data)
            elif data_type == 'dict':
                setattr(project, data_name, json.loads(data_json))
        
        # Load model results
        cursor.execute('SELECT model_name, model_data, statistics_data FROM project_models')
        
        project.model_results = {}
        project.model_statistics = {}
        
        for model_name, model_data, statistics_data in cursor.fetchall():
            project.model_results[model_name] = json.loads(model_data)
            project.model_statistics[model_name] = json.loads(statistics_data) if statistics_data else {}
        
        # Load validation results
        cursor.execute('SELECT validation_type, validation_data FROM project_validation')
        
        project.validation_results = {}
        for val_type, val_data in cursor.fetchall():
            project.validation_results[val_type] = json.loads(val_data)
        
        # Load audit log
        cursor.execute('SELECT timestamp, action, state, details FROM project_audit_log ORDER BY timestamp')
        
        project.audit_log = []
        for timestamp, action, state, details in cursor.fetchall():
            project.audit_log.append({
                'timestamp': timestamp,
                'action': action,
                'state': state,
                'details': json.loads(details) if details else {}
            })
        
        # Restore auto-checkpoint setting from loaded config
        if 'project' in configs:
            project_data = json.loads(configs['project'])
            project.config.auto_checkpoint = project_data.get('auto_checkpoint', True)
        
    logger.info(f"Project loaded from {db_path}")
    return project