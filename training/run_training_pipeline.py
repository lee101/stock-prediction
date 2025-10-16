#!/usr/bin/env python3
"""
Complete Training Pipeline with Progress Tracking and Logging
Orchestrates the entire training and validation process for all stock pairs.
"""

import sys
import os
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List
import multiprocessing as mp

# Setup comprehensive logging
def setup_logging(log_dir: Path, timestamp: str):
    """Setup comprehensive logging system"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for detailed logs
    detailed_handler = logging.FileHandler(log_dir / f'training_pipeline_{timestamp}.log')
    detailed_handler.setLevel(logging.DEBUG)
    detailed_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(detailed_handler)
    
    # Progress handler for high-level progress
    progress_handler = logging.FileHandler(log_dir / f'progress_{timestamp}.log')
    progress_handler.setLevel(logging.INFO)
    progress_handler.setFormatter(simple_formatter)
    
    # Create progress logger
    progress_logger = logging.getLogger('progress')
    progress_logger.addHandler(progress_handler)
    
    return root_logger, progress_logger


class TrainingPipelineManager:
    """Manages the complete training and validation pipeline"""
    
    def __init__(self, config_file: str = None):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.pipeline_dir = Path('pipeline_results') / self.timestamp
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger, self.progress_logger = setup_logging(
            self.pipeline_dir / 'logs', self.timestamp
        )
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Initialize components
        self.training_data_dir = Path('../trainingdata')
        self.models_dir = Path('models/per_stock')
        self.validation_dir = Path('validation_results')
        
        # Pipeline state
        self.pipeline_state = {
            'start_time': datetime.now().isoformat(),
            'symbols_to_train': [],
            'training_status': {},
            'validation_status': {},
            'overall_progress': 0.0
        }
        
        self.logger.info(f"🚀 Training Pipeline Manager initialized - {self.timestamp}")
    
    def load_config(self, config_file: str = None) -> Dict:
        """Load pipeline configuration"""
        default_config = {
            'training': {
                'episodes': 1000,
                'parallel': True,
                'validation_interval': 50,
                'save_interval': 100,
                'early_stopping_patience': 5
            },
            'validation': {
                'run_validation': True,
                'validation_threshold': 0.05  # 5% minimum return for "success"
            },
            'pipeline': {
                'auto_cleanup': True,
                'save_intermediate_results': True,
                'max_parallel_jobs': mp.cpu_count()
            }
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            # Merge configs
            for section, values in user_config.items():
                if section in default_config:
                    default_config[section].update(values)
                else:
                    default_config[section] = values
        
        # Save final config
        config_path = self.pipeline_dir / 'pipeline_config.json'
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def discover_symbols(self) -> List[str]:
        """Discover all available symbols for training"""
        train_dir = self.training_data_dir / 'train'
        test_dir = self.training_data_dir / 'test'
        
        if not train_dir.exists() or not test_dir.exists():
            self.logger.error("Training data directories not found!")
            return []
        
        # Get symbols that have both train and test data
        train_symbols = {f.stem for f in train_dir.glob('*.csv')}
        test_symbols = {f.stem for f in test_dir.glob('*.csv')}
        
        available_symbols = sorted(train_symbols & test_symbols)
        
        self.logger.info(f"📊 Discovered {len(available_symbols)} symbols with complete data:")
        for symbol in available_symbols:
            self.logger.info(f"  - {symbol}")
        
        return available_symbols
    
    def update_progress(self, message: str, progress: float = None):
        """Update pipeline progress and log"""
        if progress is not None:
            self.pipeline_state['overall_progress'] = progress
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        progress_msg = f"[{timestamp}] {message}"
        if progress is not None:
            progress_msg += f" ({progress:.1f}%)"
        
        self.progress_logger.info(progress_msg)
        self.logger.info(progress_msg)
        
        # Save state
        self.save_pipeline_state()
    
    def save_pipeline_state(self):
        """Save current pipeline state"""
        state_file = self.pipeline_dir / 'pipeline_state.json'
        with open(state_file, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2)
    
    def run_training_phase(self, symbols: List[str]) -> Dict:
        """Run the training phase for all symbols"""
        self.update_progress("🎯 Starting training phase", 10)
        
        from train_per_stock import PerStockTrainer, StockTrainingConfig
        
        # Create training config
        config = StockTrainingConfig()
        config.episodes = self.config['training']['episodes']
        config.validation_interval = self.config['training']['validation_interval']
        config.save_interval = self.config['training']['save_interval']
        
        # Initialize trainer
        trainer = PerStockTrainer(config)
        
        # Track training progress
        total_symbols = len(symbols)
        completed_symbols = 0
        
        def update_training_progress():
            nonlocal completed_symbols
            progress = 10 + (completed_symbols / total_symbols) * 60  # 10-70% for training
            self.update_progress(f"Training progress: {completed_symbols}/{total_symbols} completed", progress)
        
        try:
            if self.config['training']['parallel'] and len(symbols) > 1:
                self.logger.info(f"🔄 Running parallel training for {len(symbols)} symbols")
                
                # Use a callback to track progress
                def training_callback(result):
                    nonlocal completed_symbols
                    completed_symbols += 1
                    symbol = result.get('symbol', 'unknown')
                    success = 'error' not in result
                    self.pipeline_state['training_status'][symbol] = 'completed' if success else 'failed'
                    update_training_progress()
                
                # Parallel training with progress tracking
                with mp.Pool(processes=min(len(symbols), self.config['pipeline']['max_parallel_jobs'])) as pool:
                    results = []
                    for symbol in symbols:
                        result = pool.apply_async(trainer.train_single_stock, (symbol,), callback=training_callback)
                        results.append(result)
                    
                    # Wait for completion
                    training_results = [r.get() for r in results]
            else:
                self.logger.info(f"🔄 Running sequential training for {len(symbols)} symbols")
                training_results = []
                
                for i, symbol in enumerate(symbols):
                    self.pipeline_state['training_status'][symbol] = 'in_progress'
                    self.update_progress(f"Training {symbol} ({i+1}/{len(symbols)})")
                    
                    result = trainer.train_single_stock(symbol)
                    training_results.append(result)
                    
                    success = 'error' not in result
                    self.pipeline_state['training_status'][symbol] = 'completed' if success else 'failed'
                    completed_symbols += 1
                    update_training_progress()
            
            # Compile training summary
            successful_trainings = [r for r in training_results if 'error' not in r]
            failed_trainings = [r for r in training_results if 'error' in r]
            
            training_summary = {
                'total_symbols': len(symbols),
                'successful': len(successful_trainings),
                'failed': len(failed_trainings),
                'success_rate': len(successful_trainings) / len(symbols) if symbols else 0,
                'training_results': training_results
            }
            
            # Save training results
            training_file = self.pipeline_dir / 'training_results.json'
            with open(training_file, 'w') as f:
                json.dump(training_summary, f, indent=2)
            
            self.update_progress(f"✅ Training completed: {len(successful_trainings)}/{len(symbols)} successful", 70)
            return training_summary
            
        except Exception as e:
            self.logger.error(f"❌ Training phase failed: {e}")
            self.update_progress("❌ Training phase failed", 70)
            return {'error': str(e)}
    
    def run_validation_phase(self, symbols: List[str]) -> Dict:
        """Run the validation phase for all trained models"""
        if not self.config['validation']['run_validation']:
            self.update_progress("⏭️ Skipping validation phase", 90)
            return {'skipped': True}
        
        self.update_progress("🔍 Starting validation phase", 75)
        
        from test_validation_framework import ModelValidator
        
        # Initialize validator
        validator = ModelValidator()
        
        # Track validation progress
        total_symbols = len(symbols)
        completed_validations = 0
        
        validation_results = []
        
        for i, symbol in enumerate(symbols):
            self.pipeline_state['validation_status'][symbol] = 'in_progress'
            self.update_progress(f"Validating {symbol} ({i+1}/{len(symbols)})")
            
            try:
                metrics = validator.validate_single_model(symbol)
                if metrics:
                    validation_results.append(metrics)
                    self.pipeline_state['validation_status'][symbol] = 'completed'
                else:
                    self.pipeline_state['validation_status'][symbol] = 'failed'
                
            except Exception as e:
                self.logger.error(f"Validation failed for {symbol}: {e}")
                self.pipeline_state['validation_status'][symbol] = 'failed'
            
            completed_validations += 1
            progress = 75 + (completed_validations / total_symbols) * 15  # 75-90% for validation
            self.update_progress(f"Validation progress: {completed_validations}/{total_symbols}", progress)
        
        # Create validation summary
        validation_summary = validator.create_summary_report(validation_results)
        validation_summary['total_validated'] = len(validation_results)
        validation_summary['validation_results'] = [vars(m) for m in validation_results]
        
        # Save validation results
        validation_file = self.pipeline_dir / 'validation_results.json'
        with open(validation_file, 'w') as f:
            json.dump(validation_summary, f, indent=2)
        
        self.update_progress(f"✅ Validation completed: {len(validation_results)} models validated", 90)
        return validation_summary
    
    def generate_final_report(self, training_summary: Dict, validation_summary: Dict) -> Dict:
        """Generate comprehensive final report"""
        self.update_progress("📊 Generating final report", 95)
        
        # Calculate overall metrics
        end_time = datetime.now()
        start_time = datetime.fromisoformat(self.pipeline_state['start_time'])
        duration = (end_time - start_time).total_seconds()
        
        # Training metrics
        training_success_rate = training_summary.get('success_rate', 0)
        successful_models = training_summary.get('successful', 0)
        
        # Validation metrics
        if validation_summary.get('skipped'):
            validation_metrics = {'skipped': True}
        else:
            profitable_models = validation_summary.get('profitable_models', 0)
            avg_return = validation_summary.get('avg_return', 0)
            profitability_rate = validation_summary.get('profitability_rate', 0)
            
            validation_metrics = {
                'profitable_models': profitable_models,
                'average_return': avg_return,
                'profitability_rate': profitability_rate,
                'best_model': validation_summary.get('best_performing_model', 'N/A')
            }
        
        # Compile final report
        final_report = {
            'pipeline_info': {
                'timestamp': self.timestamp,
                'start_time': self.pipeline_state['start_time'],
                'end_time': end_time.isoformat(),
                'duration_minutes': duration / 60,
                'config': self.config
            },
            'training_summary': {
                'total_symbols': len(self.pipeline_state['symbols_to_train']),
                'successful_trainings': successful_models,
                'training_success_rate': training_success_rate
            },
            'validation_summary': validation_metrics,
            'overall_success': {
                'pipeline_completed': True,
                'models_ready_for_production': profitable_models if not validation_summary.get('skipped') else successful_models
            },
            'next_steps': self.generate_recommendations(training_summary, validation_summary)
        }
        
        # Save final report
        report_file = self.pipeline_dir / 'final_report.json'
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        # Generate human-readable summary
        self.generate_human_readable_report(final_report)
        
        return final_report
    
    def generate_recommendations(self, training_summary: Dict, validation_summary: Dict) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        success_rate = training_summary.get('success_rate', 0)
        if success_rate < 0.8:
            recommendations.append("Consider tuning hyperparameters or adjusting training configuration")
        
        if not validation_summary.get('skipped'):
            profitability_rate = validation_summary.get('profitability_rate', 0)
            if profitability_rate < 0.3:
                recommendations.append("Low profitability rate - review trading strategy and risk management")
            elif profitability_rate > 0.7:
                recommendations.append("High profitability rate - consider deploying best models to production")
            
            avg_return = validation_summary.get('avg_return', 0)
            if avg_return > 0.1:
                recommendations.append("Strong average returns - prioritize models with highest Sharpe ratios")
        
        if success_rate > 0.9 and (validation_summary.get('skipped') or validation_summary.get('profitability_rate', 0) > 0.5):
            recommendations.append("Pipeline succeeded - ready for production deployment")
        
        return recommendations
    
    def generate_human_readable_report(self, report: Dict):
        """Generate a human-readable markdown report"""
        
        report_md = f"""# Trading Pipeline Report - {self.timestamp}

## 📊 Executive Summary

**Pipeline Duration:** {report['pipeline_info']['duration_minutes']:.1f} minutes  
**Training Success Rate:** {report['training_summary']['training_success_rate']:.1%}  
**Models Ready for Production:** {report['overall_success']['models_ready_for_production']}

## 🎯 Training Results

- **Total Symbols Processed:** {report['training_summary']['total_symbols']}
- **Successful Trainings:** {report['training_summary']['successful_trainings']}
- **Training Success Rate:** {report['training_summary']['training_success_rate']:.1%}

## 🔍 Validation Results

"""
        
        if report['validation_summary'].get('skipped'):
            report_md += "**Validation was skipped as per configuration.**\n"
        else:
            val_summary = report['validation_summary']
            report_md += f"""- **Profitable Models:** {val_summary['profitable_models']}
- **Average Return:** {val_summary['average_return']:.2%}
- **Profitability Rate:** {val_summary['profitability_rate']:.1%}
- **Best Performing Model:** {val_summary['best_model']}
"""
        
        report_md += f"""
## 💡 Recommendations

"""
        for rec in report['next_steps']:
            report_md += f"- {rec}\n"
        
        report_md += f"""
## 📁 Files Generated

- Training Results: `training_results.json`
- Validation Results: `validation_results.json`
- Pipeline Config: `pipeline_config.json`
- Detailed Logs: `logs/training_pipeline_{self.timestamp}.log`
- Progress Log: `logs/progress_{self.timestamp}.log`

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save markdown report
        report_file = self.pipeline_dir / 'README.md'
        with open(report_file, 'w') as f:
            f.write(report_md)
    
    def run_complete_pipeline(self, symbols: List[str] = None) -> Dict:
        """Run the complete training and validation pipeline"""
        
        try:
            # Discover symbols if not provided
            if symbols is None:
                symbols = self.discover_symbols()
            
            if not symbols:
                raise ValueError("No symbols available for training")
            
            self.pipeline_state['symbols_to_train'] = symbols
            self.update_progress(f"🎯 Pipeline started with {len(symbols)} symbols", 5)
            
            # Phase 1: Training
            training_summary = self.run_training_phase(symbols)
            if 'error' in training_summary:
                raise Exception(f"Training phase failed: {training_summary['error']}")
            
            # Phase 2: Validation
            validation_summary = self.run_validation_phase(symbols)
            
            # Phase 3: Final Report
            final_report = self.generate_final_report(training_summary, validation_summary)
            
            self.update_progress("🎉 Pipeline completed successfully!", 100)
            
            # Print summary to console
            self.print_pipeline_summary(final_report)
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            self.update_progress(f"❌ Pipeline failed: {e}", None)
            
            error_report = {
                'pipeline_info': {'timestamp': self.timestamp},
                'error': str(e),
                'pipeline_completed': False
            }
            
            error_file = self.pipeline_dir / 'error_report.json'
            with open(error_file, 'w') as f:
                json.dump(error_report, f, indent=2)
            
            return error_report
    
    def print_pipeline_summary(self, report: Dict):
        """Print a concise summary to console"""
        print("\n" + "="*60)
        print(f"🎉 TRAINING PIPELINE COMPLETED - {self.timestamp}")
        print("="*60)
        
        print(f"⏱️  Duration: {report['pipeline_info']['duration_minutes']:.1f} minutes")
        print(f"📈 Training Success: {report['training_summary']['successful_trainings']}/{report['training_summary']['total_symbols']} symbols")
        
        if not report['validation_summary'].get('skipped'):
            val = report['validation_summary']
            print(f"💰 Profitable Models: {val['profitable_models']}")
            print(f"📊 Average Return: {val['average_return']:.2%}")
            print(f"🏆 Best Model: {val['best_model']}")
        
        print(f"🚀 Models Ready: {report['overall_success']['models_ready_for_production']}")
        print(f"📁 Results saved to: {self.pipeline_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Run complete training pipeline')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to train')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--episodes', type=int, help='Training episodes override')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel training')
    parser.add_argument('--no-validation', action='store_true', help='Skip validation phase')
    
    args = parser.parse_args()
    
    # Create pipeline manager
    pipeline = TrainingPipelineManager(config_file=args.config)
    
    # Override config with command line args
    if args.episodes:
        pipeline.config['training']['episodes'] = args.episodes
    if args.no_parallel:
        pipeline.config['training']['parallel'] = False
    if args.no_validation:
        pipeline.config['validation']['run_validation'] = False
    
    # Run pipeline
    results = pipeline.run_complete_pipeline(symbols=args.symbols)
    
    # Exit with appropriate code
    if results.get('pipeline_completed', False):
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()