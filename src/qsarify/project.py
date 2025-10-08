"""
Project management system for QSAR/QSPR modeling workflows.

This module implements a finite state machine-based Project class that manages
the complete modeling workflow from data loading through validation and visualization.
"""

import logging
import sqlite3
import uuid
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from .preprocessing.preprocessing import PreprocessingConfig, comprehensive_preprocess
from .modeling.regressors import UnifiedModelComparison, OptimizationConfig
from .modeling.ga import GeneticConfig
from .validation.procedures import ValidationResult, leave_many_out_cv, y_scrambling
from .exceptions import QSARifyError


class ProjectState(Enum):
    """Strict finite state machine states for project workflow."""
    INITIALIZED = auto()    # Project created, no data loaded
    DATA_LOADED = auto()    # Raw data loaded and validated  
    PREPROCESSED = auto()   # Data preprocessed and ready for modeling
    MODELED = auto()        # Models trained and evaluated
    VALIDATED = auto()      # Additional validation procedures completed
    COMPLETED = auto()      # All analysis complete, ready for export


class WarningHandler(ABC):
    """Abstract base class for handling destructive operation warnings."""
    
    @abstractmethod
    def handle_warning(self, operation: str, affected_data: List[str], project_state: str) -> bool:
        """
        Handle a destructive operation warning.
        
        Args:
            operation: Name of the destructive operation
            affected_data: List of data that will be lost
            project_state: Current project state
            
        Returns:
            True if operation should proceed, False otherwise
        """
        pass


class InteractiveWarningHandler(WarningHandler):
    """Interactive warning handler that prompts user for confirmation."""
    
    def handle_warning(self, operation: str, affected_data: List[str], project_state: str) -> bool:
        print(f"\n⚠️  WARNING: Operation '{operation}' will delete existing data:")
        for item in affected_data:
            print(f"  - {item}")
        print(f"Current project state: {project_state}")
        
        while True:
            response = input("Do you want to continue? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")


class AutoConfirmWarningHandler(WarningHandler):
    """Warning handler that automatically confirms all operations."""
    
    def handle_warning(self, operation: str, affected_data: List[str], project_state: str) -> bool:
        return True


class BlockingWarningHandler(WarningHandler):
    """Warning handler that blocks all destructive operations."""
    
    def handle_warning(self, operation: str, affected_data: List[str], project_state: str) -> bool:
        return False


@dataclass
class ProjectConfig:
    """Configuration settings for the entire project."""
    name: str
    description: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    preprocessing_config: Optional[PreprocessingConfig] = None
    optimization_config: Optional[OptimizationConfig] = None
    genetic_config: Optional[GeneticConfig] = None
    auto_checkpoint: bool = True
    log_level: str = "INFO"
    warning_handler: Optional[WarningHandler] = field(default_factory=lambda: InteractiveWarningHandler())


class ProjectError(QSARifyError):
    """Base exception for project-related errors."""
    pass


class InvalidStateTransitionError(ProjectError):
    """Raised when attempting an invalid state transition."""
    pass


class DestructiveOperationError(ProjectError):
    """Raised when attempting a destructive operation without confirmation."""
    pass


class Project:
    """
    Main project class implementing finite state machine for QSAR/QSPR workflows.
    
    This class manages the complete modeling workflow:
    1. DATA_LOADED: Load and validate raw data
    2. PREPROCESSED: Apply preprocessing pipeline  
    3. MODELED: Train and evaluate models
    4. VALIDATED: Additional validation procedures
    5. COMPLETED: Export results and plots
    
    Features:
    - Finite state machine with validation
    - Automatic checkpointing
    - Audit logging
    - SQLite persistence
    - Destructive operation warnings
    """
    
    def __init__(self, config: ProjectConfig, project_file: Optional[Union[str, Path]] = None):
        """
        Initialize a new project.
        
        Args:
            config: Project configuration
            project_file: Optional path to project file for persistence
        """
        self.config = config
        self.project_id = str(uuid.uuid4())
        self.state = ProjectState.INITIALIZED
        self.project_file = Path(project_file) if project_file else None
        
        # Data containers
        self.raw_data: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.feature_columns: Optional[List[str]] = None
        
        # Processed data
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.preprocessing_info: Optional[Dict[str, Any]] = None
        self.scaler: Optional[Any] = None
        
        # Model results
        self.model_comparison: Optional[UnifiedModelComparison] = None
        self.model_results: Dict[str, Any] = {}
        self.model_statistics: Dict[str, Any] = {}
        
        # Validation results
        self.validation_results: Dict[str, ValidationResult] = {}
        
        # Audit log
        self.audit_log: List[Dict[str, Any]] = []
        
        # Setup logging
        self._setup_logging()
        
        # Log project creation
        self._log_action("project_created", {
            "project_id": self.project_id,
            "name": config.name,
            "state": self.state.name
        })
        
        # Auto-checkpoint if enabled
        if config.auto_checkpoint and self.project_file:
            self._checkpoint()
    
    def _setup_logging(self):
        """Setup project-specific logging."""
        self.logger = logging.getLogger(f"qsarify.project.{self.config.name}")
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Create handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _log_action(self, action: str, details: Dict[str, Any] = None):
        """Log a user action for audit purposes."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "state": self.state.name,
            "details": details or {}
        }
        self.audit_log.append(log_entry)
        self.logger.info(f"Action: {action} | State: {self.state.name}")
    
    def _validate_state_transition(self, target_state: ProjectState, action: str = ""):
        """Validate if a state transition is allowed in strict FSM."""
        # Strict FSM: mostly linear progression with limited back-transitions
        valid_transitions = {
            ProjectState.INITIALIZED: [ProjectState.DATA_LOADED],
            ProjectState.DATA_LOADED: [ProjectState.PREPROCESSED],
            ProjectState.PREPROCESSED: [ProjectState.MODELED],
            ProjectState.MODELED: [ProjectState.VALIDATED, ProjectState.COMPLETED],
            ProjectState.VALIDATED: [ProjectState.COMPLETED],
            ProjectState.COMPLETED: []  # Terminal state - no transitions allowed
        }
        
        # Allow limited back-transitions for re-doing steps (destructive operations)
        destructive_back_transitions = {
            (ProjectState.PREPROCESSED, ProjectState.DATA_LOADED): "load_data",
            (ProjectState.MODELED, ProjectState.PREPROCESSED): "preprocess", 
            (ProjectState.VALIDATED, ProjectState.MODELED): "run_models",
            (ProjectState.COMPLETED, ProjectState.MODELED): "run_models",
            (ProjectState.COMPLETED, ProjectState.VALIDATED): "run_validation"
        }
        
        # Check normal forward transitions
        if target_state in valid_transitions.get(self.state, []):
            return
        
        # Check destructive back-transitions
        transition_key = (self.state, target_state)
        if transition_key in destructive_back_transitions:
            expected_action = destructive_back_transitions[transition_key]
            if action == expected_action:
                return  # Allow this destructive back-transition
        
        # Invalid transition
        raise InvalidStateTransitionError(
            f"Cannot transition from {self.state.name} to {target_state.name} "
            f"for action: {action}. Strict FSM allows only linear progression "
            f"with limited destructive back-transitions."
        )
    
    def _check_destructive_operation(self, operation: str):
        """Check if operation is destructive and handle using configured warning handler."""
        destructive_ops = {
            "load_data": ["preprocessed data", "models", "validation results"],
            "preprocess": ["models", "validation results"], 
            "run_models": ["validation results"]
        }
        
        if operation in destructive_ops:
            affected = destructive_ops[operation]
            
            # Use the configured warning handler
            if self.config.warning_handler:
                confirmed = self.config.warning_handler.handle_warning(
                    operation, affected, self.state.name
                )
            else:
                # Fallback to requiring explicit confirmation
                confirmed = False
            
            if not confirmed:
                raise DestructiveOperationError(
                    f"Operation '{operation}' cancelled. Would delete: {', '.join(affected)}"
                )
            
            # Log destructive operation
            self._log_action(f"destructive_operation_{operation}", {
                "affected_data": affected,
                "confirmed": confirmed
            })
    
    def _checkpoint(self):
        """Save current project state to file."""
        if not self.project_file:
            self.logger.warning("No project file specified, skipping checkpoint")
            return
            
        try:
            from .io.persistence import save_project
            save_project(self, self.project_file)
            self.logger.info(f"Checkpoint saved to {self.project_file}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def load_data(self, 
                  data: pd.DataFrame, 
                  target_column: str,
                  feature_columns: Optional[List[str]] = None) -> None:
        """
        Load raw data into the project.
        
        Args:
            data: Raw dataset
            target_column: Name of target variable column
            feature_columns: Optional list of feature columns (if None, use all except target)
        """
        # Check if destructive
        if self.state != ProjectState.INITIALIZED:
            self._check_destructive_operation("load_data")
        
        # Validate data
        if target_column not in data.columns:
            raise ProjectError(f"Target column '{target_column}' not found in data")
        
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        else:
            missing_cols = set(feature_columns) - set(data.columns)
            if missing_cols:
                raise ProjectError(f"Feature columns not found: {missing_cols}")
        
        # Clear downstream results if reloading
        if self.state != ProjectState.INITIALIZED:
            self._clear_downstream_results()
        
        # Store data
        self.raw_data = data.copy()
        self.target_column = target_column
        self.feature_columns = feature_columns
        
        # Transition state
        self.state = ProjectState.DATA_LOADED
        
        # Log action
        self._log_action("data_loaded", {
            "n_samples": len(data),
            "n_features": len(feature_columns),
            "target_column": target_column
        })
        
        self.logger.info(f"Data loaded: {len(data)} samples, {len(feature_columns)} features")
        
        # Auto-checkpoint
        if self.config.auto_checkpoint:
            self._checkpoint()
    
    def preprocess(self, 
                   preprocessing_config: Optional[PreprocessingConfig] = None) -> None:
        """
        Apply preprocessing pipeline to loaded data.
        
        Args:
            preprocessing_config: Preprocessing configuration (uses project default if None)
        """
        # Validate state
        self._validate_state_transition(ProjectState.PREPROCESSED, "preprocess")
        
        # Check if destructive (when re-preprocessing from later states)
        if self.state != ProjectState.DATA_LOADED:
            self._check_destructive_operation("preprocess")
        
        if self.raw_data is None:
            raise ProjectError("No data loaded. Call load_data() first.")
        
        # Use provided config or project default
        if preprocessing_config is None:
            if self.config.preprocessing_config is None:
                preprocessing_config = PreprocessingConfig()
            else:
                preprocessing_config = self.config.preprocessing_config
        
        # Clear downstream results if re-preprocessing
        if self.state != ProjectState.DATA_LOADED:
            self._clear_downstream_results(keep_preprocessing=False)
        
        # Apply preprocessing
        X = self.raw_data[self.feature_columns]
        y = self.raw_data[self.target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test, self.scaler, self.preprocessing_info = \
            comprehensive_preprocess(X, y, preprocessing_config)
        
        # Store config used
        self.config.preprocessing_config = preprocessing_config
        
        # Transition state
        self.state = ProjectState.PREPROCESSED
        
        # Log action
        self._log_action("data_preprocessed", {
            "train_samples": len(self.X_train),
            "test_samples": len(self.X_test),
            "final_features": self.preprocessing_info.get('n_final_features', 0)
        })
        
        self.logger.info(f"Preprocessing complete: {len(self.X_train)} train, {len(self.X_test)} test samples")
        
        # Auto-checkpoint
        if self.config.auto_checkpoint:
            self._checkpoint()
    
    def run_models(self, 
                   optimization_config: Optional[OptimizationConfig] = None,
                   genetic_config: Optional[GeneticConfig] = None,
                   include_models: Optional[List[str]] = None) -> None:
        """
        Train and evaluate models.
        
        Args:
            optimization_config: Model optimization configuration
            genetic_config: Genetic algorithm configuration
            include_models: List of model types to include
        """
        # Validate state
        self._validate_state_transition(ProjectState.MODELED, "run_models")
        
        # Check if destructive (when re-running models from later states)
        if self.state != ProjectState.PREPROCESSED:
            self._check_destructive_operation("run_models")
        
        if self.X_train is None:
            raise ProjectError("No preprocessed data. Call preprocess() first.")
        
        # Use provided configs or project defaults
        if optimization_config is None:
            optimization_config = self.config.optimization_config or OptimizationConfig()
        if genetic_config is None:
            genetic_config = self.config.genetic_config or GeneticConfig()
        
        # Clear downstream results if re-running
        if self.state != ProjectState.PREPROCESSED:
            self._clear_downstream_results(keep_preprocessing=True, keep_models=False)
        
        # Setup model comparison
        self.model_comparison = UnifiedModelComparison(
            optimization_config=optimization_config,
            genetic_config=genetic_config,
            include_models=include_models
        )
        
        # Run comprehensive comparison
        self.model_results = self.model_comparison.run_comprehensive_comparison(
            self.raw_data[self.feature_columns],
            self.raw_data[self.target_column],
            self.config.preprocessing_config
        )
        
        self.model_statistics = self.model_comparison.statistics
        
        # Store configs used
        self.config.optimization_config = optimization_config
        self.config.genetic_config = genetic_config
        
        # Transition state
        self.state = ProjectState.MODELED
        
        # Log action
        successful_models = [name for name, result in self.model_results.items() 
                           if "error" not in result]
        self._log_action("models_trained", {
            "total_models": len(self.model_results),
            "successful_models": len(successful_models),
            "model_names": successful_models
        })
        
        self.logger.info(f"Models trained: {len(successful_models)}/{len(self.model_results)} successful")
        
        # Auto-checkpoint
        if self.config.auto_checkpoint:
            self._checkpoint()
    
    def run_validation(self, 
                       validation_types: List[str] = None,
                       validation_params: Dict[str, Dict[str, Any]] = None) -> None:
        """
        Run additional validation procedures.
        
        Args:
            validation_types: List of validation types ['leave_many_out', 'y_scrambling']
            validation_params: Parameters for each validation type
        """
        # Validate state
        self._validate_state_transition(ProjectState.VALIDATED, "run_validation")
        
        if not self.model_results:
            raise ProjectError("No models available. Call run_models() first.")
        
        if validation_types is None:
            validation_types = ['leave_many_out', 'y_scrambling']
        
        if validation_params is None:
            validation_params = {}
        
        # Get best model for validation
        best_model_name, best_model = self.model_comparison.get_best_model()
        
        # Run requested validations
        for val_type in validation_types:
            if val_type == 'leave_many_out':
                params = validation_params.get('leave_many_out', {})
                result = leave_many_out_cv(
                    best_model.model_,
                    pd.concat([self.X_train, self.X_test]),
                    pd.concat([self.y_train, self.y_test]),
                    model_name=best_model_name,
                    **params
                )
                self.validation_results['leave_many_out'] = result
                
            elif val_type == 'y_scrambling':
                params = validation_params.get('y_scrambling', {})
                result = y_scrambling(
                    best_model.model_,
                    self.X_train,
                    self.y_train,
                    model_name=best_model_name,
                    **params
                )
                self.validation_results['y_scrambling'] = result
        
        # Transition state
        self.state = ProjectState.VALIDATED
        
        # Log action
        self._log_action("validation_completed", {
            "validation_types": validation_types,
            "best_model": best_model_name
        })
        
        self.logger.info(f"Validation completed: {validation_types}")
        
        # Auto-checkpoint
        if self.config.auto_checkpoint:
            self._checkpoint()
    
    def _clear_downstream_results(self, keep_preprocessing: bool = True, keep_models: bool = True):
        """Clear results that are invalidated by upstream changes."""
        if not keep_preprocessing:
            self.X_train = None
            self.X_test = None 
            self.y_train = None
            self.y_test = None
            self.preprocessing_info = None
            self.scaler = None
        
        if not keep_models:
            self.model_comparison = None
            self.model_results = {}
            self.model_statistics = {}
        
        # Always clear validation results
        self.validation_results = {}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current project state."""
        summary = {
            "project_id": self.project_id,
            "name": self.config.name,
            "state": self.state.name,
            "created_at": self.config.created_at.isoformat(),
            "data_info": {},
            "model_info": {},
            "validation_info": {}
        }
        
        # Data information
        if self.raw_data is not None:
            summary["data_info"] = {
                "n_samples": len(self.raw_data),
                "n_features": len(self.feature_columns) if self.feature_columns else 0,
                "target_column": self.target_column
            }
        
        # Model information
        if self.model_results:
            successful_models = [name for name, result in self.model_results.items() 
                               if "error" not in result]
            summary["model_info"] = {
                "total_models": len(self.model_results),
                "successful_models": len(successful_models),
                "model_names": successful_models
            }
            
            if successful_models and self.model_comparison:
                best_name, _ = self.model_comparison.get_best_model()
                summary["model_info"]["best_model"] = best_name
        
        # Validation information
        if self.validation_results:
            summary["validation_info"] = {
                "completed_validations": list(self.validation_results.keys())
            }
        
        return summary
    
    def save(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save project to file.
        
        Args:
            file_path: Optional file path (uses project_file if None)
        """
        path_to_use = Path(file_path) if file_path else self.project_file
        if not path_to_use:
            raise ProjectError("No file path specified and no project_file set")
        
        from .io.persistence import save_project
        save_project(self, path_to_use)
        
        # Update project_file if new path provided
        if file_path:
            self.project_file = Path(file_path)
    
    def export_summary(self) -> str:
        """Export a text summary of the project."""
        summary = self.get_summary()
        
        lines = [
            f"QSAR/QSPR Project Summary",
            f"=" * 50,
            f"Name: {summary['name']}",
            f"State: {summary['state']}",
            f"Created: {summary['created_at'][:19]}",  # Remove microseconds
            f"ID: {summary['project_id']}",
            "",
            "Data Information:",
        ]
        
        if summary['data_info']:
            data = summary['data_info']
            lines.extend([
                f"  - Samples: {data.get('n_samples', 'N/A')}",
                f"  - Features: {data.get('n_features', 'N/A')}",
                f"  - Target: {data.get('target_column', 'N/A')}"
            ])
        else:
            lines.append("  - No data loaded")
            
        lines.append("")
        lines.append("Model Information:")
        
        if summary['model_info']:
            model = summary['model_info']
            lines.extend([
                f"  - Total models: {model.get('total_models', 0)}",
                f"  - Successful: {model.get('successful_models', 0)}",
                f"  - Best model: {model.get('best_model', 'N/A')}"
            ])
        else:
            lines.append("  - No models trained")
            
        lines.append("")
        lines.append("Validation Information:")
        
        if summary['validation_info']:
            val = summary['validation_info']
            lines.append(f"  - Completed: {', '.join(val.get('completed_validations', []))}")
        else:
            lines.append("  - No validation completed")
            
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation of the project."""
        return f"Project(name='{self.config.name}', state={self.state.name}, id={self.project_id[:8]}...)"


# Factory functions

def create_project(name: str, 
                  description: str = "",
                  author: str = "",
                  project_file: Optional[Union[str, Path]] = None,
                  warning_handler: Optional[WarningHandler] = None) -> Project:
    """
    Create a new project with default configuration.
    
    Args:
        name: Project name
        description: Project description
        author: Project author
        project_file: Optional path for project persistence
        warning_handler: Optional warning handler (defaults to InteractiveWarningHandler)
        
    Returns:
        New Project instance
    """
    config = ProjectConfig(
        name=name,
        description=description, 
        author=author,
        warning_handler=warning_handler or InteractiveWarningHandler()
    )
    
    return Project(config, project_file)


def load_project_from_file(file_path: Union[str, Path]) -> Project:
    """
    Load project from file.
    
    Args:
        file_path: Path to project file
        
    Returns:
        Loaded Project instance
    """
    from .io.persistence import load_project
    return load_project(file_path)


def create_batch_project(name: str,
                        project_file: Optional[Union[str, Path]] = None) -> Project:
    """
    Create project configured for batch/non-interactive use.
    
    Args:
        name: Project name
        project_file: Optional path for project persistence
        
    Returns:
        Project configured with AutoConfirmWarningHandler
    """
    return create_project(
        name=name,
        project_file=project_file,
        warning_handler=AutoConfirmWarningHandler()
    )