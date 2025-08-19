import os
import argparse
import logging
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import yaml
import psycopg2
from psycopg2.extras import Json, RealDictCursor

# Константы
DEFAULT_LOG_PATH = '/var/log/mad_trainer.log'
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3
DEFAULT_CONFIG_DIR = '/app/config'
DEFAULT_CONFIG_PATH = f'{DEFAULT_CONFIG_DIR}/default_config.yaml'

# Имена переменных среды
ENV_VICTORIAMETRICS_URL = 'VICTORIAMETRICS_URL'
ENV_DB_CONN_STRING = 'DB_CONN_STRING'
ENV_LOG_PATH = 'LOG_PATH'
ENV_USER_CONFIG_PATH = 'CONFIG_PATH'

class MadTrainerDatabaseManager:
    """Сервис управления базой данных для системы обучения моделей обнаружения аномалий."""

    def __init__(self, config_path: Optional[str] = None):
        """Инициализация сервиса."""
        self._setup_logging()
        self.config = self._load_config(config_path)
        self._validate_config()
        self.db_conn = self._init_db_connection()
        self.logger.info("Сервис управления БД MAD Trainer инициализирован")

    def _setup_logging(self) -> None:
        """Настройка системы логирования."""
        log_path = os.getenv(ENV_LOG_PATH, DEFAULT_LOG_PATH)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        self.logger = logging.getLogger('MadTrainerDB')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Файловый обработчик с ротацией
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        
        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Загрузка конфигурации с дефолтами."""
        try:
            # Создаем директорию для конфигов если нужно
            os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
            
            # Определяем путь к пользовательскому конфигу
            user_config_path = config_path or os.getenv(ENV_USER_CONFIG_PATH, f'{DEFAULT_CONFIG_DIR}/mad-trainer-config.yaml')
            
            if not Path(user_config_path).exists():
                raise FileNotFoundError(f"Конфигурационный файл не найден: {user_config_path}")
            
            with open(user_config_path) as f:
                config = yaml.safe_load(f) or {}
            
            # Проверяем наличие обязательных секций
            if 'general' not in config:
                config['general'] = {}
                
            # Добавляем параметры из переменных среды
            if ENV_VICTORIAMETRICS_URL in os.environ:
                config['general']['victoriametrics_url'] = os.environ[ENV_VICTORIAMETRICS_URL]
                
            if ENV_DB_CONN_STRING in os.environ:
                config['general']['db_conn_string'] = os.environ[ENV_DB_CONN_STRING]
                
            if ENV_LOG_PATH in os.environ:
                config['general']['log_path'] = os.environ[ENV_LOG_PATH]
                
            return config
        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации: {str(e)}")
            raise

    def _validate_config(self) -> None:
        """Проверка конфигурации."""
        general = self.config.get('general', {})
        
        if not general.get('victoriametrics_url'):
            raise ValueError(f"Необходимо указать victoriametrics_url в конфигурации или переменной среды {ENV_VICTORIAMETRICS_URL}")
            
        if not general.get('db_conn_string'):
            raise ValueError(f"Необходимо указать db_conn_string в конфигурации или переменной среды {ENV_DB_CONN_STRING}")
            
        if not general.get('metrics'):
            self.logger.warning("Конфигурация не содержит метрик")
            
        if not general.get('models'):
            self.logger.warning("Конфигурация не содержит моделей")

    def _init_db_connection(self) -> 'psycopg2.connection':
        """Установка соединения с PostgreSQL."""
        conn_string = self.config['general'].get('db_conn_string')
        if not conn_string:
            raise ValueError(f"Необходимо указать db_conn_string в конфигурации или переменной среды {ENV_DB_CONN_STRING}")
        
        try:
            conn = psycopg2.connect(**self._parse_db_conn_string(conn_string))
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError as e:
            self.logger.error(f"Ошибка подключения к БД: {str(e)}")
            raise

    def _parse_db_conn_string(self, conn_string: str) -> Dict:
        """Парсинг строки подключения к PostgreSQL."""
        try:
            if '://' in conn_string:
                conn_string = conn_string.split('://')[1]
            
            if '@' in conn_string:
                user_part, host_part = conn_string.split('@', 1)
                user, password = user_part.split(':', 1)
            else:
                host_part = conn_string
                user = password = None
            
            if '/' in host_part:
                host_port, database = host_part.split('/', 1)
            else:
                host_port = host_part
                database = None
            
            if ':' in host_port:
                host, port = host_port.split(':', 1)
            else:
                host = host_port
                port = DEFAULT_DB_PORT
            
            return {
                'host': host,
                'database': database,
                'user': user,
                'password': password,
                'port': port
            }
        except Exception as e:
            raise ValueError(f"Неверный формат строки подключения: {str(e)}")

    def _parse_duration(self, duration_str: str) -> int:
        """Преобразование строки длительности в секунды."""
        try:
            if duration_str.endswith('d'):
                return int(duration_str[:-1]) * 86400
            elif duration_str.endswith('h'):
                return int(duration_str[:-1]) * 3600
            elif duration_str.endswith('m'):
                return int(duration_str[:-1]) * 60
            elif duration_str.endswith('s'):
                return int(duration_str[:-1])
            return int(duration_str)
        except ValueError:
            raise ValueError(f"Неверный формат длительности: {duration_str}")

    def _insert_exclude_periods(self, cursor: 'psycopg2.cursor', metric_id: int, metric_config: Dict) -> None:
        """Добавление аномальных периодов для метрик в систему аномалий."""
        if 'exclude_periods' not in metric_config:
            self.logger.debug(f"Для метрики ID {metric_id} нет аномальных периодов")
            return
            
        for period in metric_config['exclude_periods']:
            try:
                # Валидация обязательных полей
                if not all(key in period for key in ['start', 'end', 'reason', 'anomaly_type']):
                    self.logger.error(f"Не все обязательные параметры указаны в аномальном периоде для метрики ID {metric_id}")
                    continue
                    
                # Преобразование времени
                start_time = datetime.fromisoformat(period['start'].replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(period['end'].replace('Z', '+00:00'))
                
                # Проверка на существование периода
                cursor.execute(
                    """SELECT id FROM anomaly_system 
                    WHERE metric_id = %s AND start_time = %s AND end_time = %s""",
                    (metric_id, start_time, end_time)
                )
                
                if cursor.fetchone() is None:
                    cursor.execute(
                        """
                        INSERT INTO anomaly_system (
                            start_time, 
                            end_time, 
                            anomaly_type, 
                            average_anom_score, 
                            metric_id, 
                            description
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            start_time, 
                            end_time,
                            period['anomaly_type'],
                            100,
                            metric_id,
                            period['reason']
                        )
                    )
                    self.logger.info(
                        f"Добавлен аномальный период для метрики {metric_id}: "
                        f"{start_time} - {end_time} ({period['reason']})"
                    )
                    
            except ValueError as e:
                self.logger.error(f"Ошибка формата времени в аномальном периоде: {str(e)}")
            except Exception as e:
                self.logger.error(f"Ошибка добавления аномального периода: {str(e)}")


    def convert_to_seconds(self, time_str: str) -> int:
        """Конвертирует строку с временным интервалом в количество секунд.
        
        Поддерживает различные форматы:
        - '30s' - 30 секунд
        - '1m' - 1 минута (60 секунд)
        - '2h' - 2 часа (7200 секунд)
        - '1d' - 1 день (86400 секунд)
        - '1w' - 1 неделя (604800 секунд)
        - Число без суффикса - интерпретируется как секунды
        
        Args:
            time_str: Строка с временным интервалом (например '1m', '30s')
            
        Returns:
            int: Количество секунд
            
        Raises:
            ValueError: Если формат строки некорректен
        """
        self.logger.debug(f"Конвертация временного интервала: {time_str}")
        
        if not time_str:
            self.logger.error("Получена пустая строка временного интервала")
            raise ValueError("Пустая строка временного интервала")
        
        # Если строка состоит только из цифр, интерпретируем как секунды
        if time_str.isdigit():
            seconds = int(time_str)
            self.logger.debug(f"Интервал в секундах: {seconds} (без суффикса)")
            return seconds
        
        try:
            # Извлекаем числовую часть и единицу измерения
            num = int(time_str[:-1])
            unit = time_str[-1].lower()
        except (ValueError, IndexError):
            error_msg = f"Некорректный формат временного интервала: {time_str}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Словарь конвертации единиц в секунды
        conversion = {
            's': 1,                # секунды
            'm': 60,               # минуты
            'h': 60 * 60,          # часы
            'd': 24 * 60 * 60,     # дни
            'w': 7 * 24 * 60 * 60  # недели
        }
        
        if unit not in conversion:
            error_msg = f"Неизвестная единица измерения: {unit}. Допустимы: s,m,h,d,w"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        seconds = num * conversion[unit]
        self.logger.debug(f"Конвертировано: {time_str} = {seconds} секунд")
        
        return seconds
        
    def init_database(self) -> None:
        """Инициализация структуры базы данных."""
        if not self.config['general'].get('metrics') or not self.config['general'].get('models'):
            raise ValueError("Конфигурация должна содержать метрики и модели")
        
        self.logger.info("Начало инициализации БД MAD Trainer")
        
        try:
            with self.db_conn.cursor() as cursor:
                
                # Добавление метрик
                for metric in self.config['general']['metrics']:
                    cursor.execute(
                        """
                        INSERT INTO metrics (name, status, query)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (name) DO UPDATE 
                        SET query = EXCLUDED.query
                        RETURNING id
                        """,
                        (
                            metric['name'], 
                            'active',
                            metric['query']
                        )
                    )
                    metric_id = cursor.fetchone()[0]
                    self._insert_exclude_periods(cursor, metric_id, metric)
                    self.logger.info(f"Добавлена метрика {metric['name']} (ID: {metric_id})")
                
                # Добавление информации о моделях
                for model in self.config['general']['models']:
                    # Получаем ID основной метрики
                    cursor.execute(
                        "SELECT id FROM metrics WHERE name = %s",
                        (model['main_metric'],)
                    )
                    metric_id = cursor.fetchone()[0]
                    
                    # Получаем параметры обучения
                    training_period = model.get('training_period', {})
                    if 'fixed_range' in training_period:
                        training_start = training_period['fixed_range']['start']
                        training_end = training_period['fixed_range']['end']
                        step = self.convert_to_seconds(training_period['fixed_range'].get('step'))
                    else:
                        lookback_str = training_period['auto_range']['lookback_period']
                        seconds = self.convert_to_seconds(lookback_str)
                        days = seconds / 86400  # Convert seconds to days
                        training_start = datetime.now() - timedelta(days=days)
                        training_end = datetime.now()
                        step = self.convert_to_seconds(training_period['auto_range'].get('step'))
                    
                    # Добавляем информацию о модели
                    cursor.execute(
                        """
                        INSERT INTO models_info (
                            name, metric_id, max_stored_versions, 
                            hyperparams_mode, active_version,
                            training_start, training_end, step
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (name) DO UPDATE
                        SET metric_id = EXCLUDED.metric_id,
                            max_stored_versions = EXCLUDED.max_stored_versions,
                            hyperparams_mode = EXCLUDED.hyperparams_mode,
                            training_start = EXCLUDED.training_start,
                            training_end = EXCLUDED.training_end,
                            step = EXCLUDED.step
                        RETURNING id
                        """,
                        (
                            model['name'],
                            metric_id,
                            model['version_history'],
                            model['hyperparameter_mode'],
                            None,
                            training_start,
                            training_end,
                            step
                        )
                    )
                    model_info_id = cursor.fetchone()[0]
                    
                    # Добавляем дополнительные метрики (features)
                    for metric_name in model.get('additional_metrics', []):
                        cursor.execute(
                            "SELECT id FROM metrics WHERE name = %s",
                            (metric_name,)
                        )
                        additional_metric_id = cursor.fetchone()[0]
                        
                        cursor.execute(
                            """
                            INSERT INTO features (model_id, metric_id)
                            VALUES (%s, %s)
                            ON CONFLICT DO NOTHING
                            """,
                            (model_info_id, additional_metric_id)
                        )
                    
                    # Добавляем начальную версию модели
                    hyperparams = self._get_hyperparams(model)
                    cursor.execute(
                        """
                        INSERT INTO models (
                            model_data, created_at, status,
                            version, model_id, hyperparams
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            None,  # model_data будет добавлен после обучения
                            datetime.now(),
                            'waiting',
                            '1.0',
                            model_info_id,
                            Json(hyperparams)
                        )
                    )
                    
                    self.logger.info(f"Добавлена модель {model['name']} (ID: {model_info_id})")
                
                self.db_conn.commit()
                self.logger.info("Инициализация БД MAD Trainer успешно завершена")
                
        except Exception as e:
            self.db_conn.rollback()
            self.logger.error(f"Ошибка инициализации БД: {str(e)}")
            raise

    def _get_hyperparams(self, model_config: Dict) -> Dict:
        """Получение гиперпараметров модели."""
        if model_config['hyperparameter_mode'] == 'manual':
            return model_config['manual_params']
        
        # Параметры по умолчанию для optuna
        return {
            'direction': 'minimize',
            'metric': 'val_loss',
            'n_trials': 20,
            'sampler': 'TPE',
            'pruner': 'Hyperband',
            'fixed_parameters': {
                'loss': 'mean_squared_error',
                'optimizer': 'adam',
                'validation_split': 0.2
            }
        }

    def clean_old_versions(self) -> None:
        """Очистка старых версий моделей с учетом history_limit."""
        self.logger.info("Начало очистки старых версий моделей")
        
        try:
            with self.db_conn.cursor() as cursor:
                # Получаем все модели с их лимитами версий
                cursor.execute("""
                    SELECT m.id, m.name, m.version_history
                    FROM models m
                """)
                models = cursor.fetchall()
                
                if not models:
                    self.logger.warning("Нет моделей для очистки версий")
                    return
                
                total_deleted = 0
                
                for model in models:
                    model_id, model_name, version_history = model
                    
                    # Получаем количество текущих версий
                    cursor.execute(
                        "SELECT COUNT(*) FROM model_versions WHERE model_id = %s",
                        (model_id,)
                    )
                    current_versions = cursor.fetchone()[0]
                    
                    if current_versions <= version_history:
                        self.logger.debug(
                            f"Для модели {model_name} не требуется очистка "
                            f"(текущие версии: {current_versions}, лимит: {version_history})"
                        )
                        continue
                    
                    # Определяем сколько версий нужно удалить
                    to_delete = current_versions - version_history
                    
                    # Получаем ID старых версий для удаления
                    cursor.execute("""
                        SELECT id FROM model_versions
                        WHERE model_id = %s
                        ORDER BY created_at ASC
                        LIMIT %s
                    """, (model_id, to_delete))
                    
                    old_version_ids = [row[0] for row in cursor.fetchall()]
                    
                    # Удаляем старые версии
                    cursor.execute(
                        "DELETE FROM model_versions WHERE id = ANY(%s)",
                        (old_version_ids,)
                    )
                    
                    deleted = cursor.rowcount
                    total_deleted += deleted
                    
                    self.logger.info(
                        f"Для модели {model_name} удалено {deleted} старых версий "
                        f"(осталось: {version_history})"
                    )
                
                self.db_conn.commit()
                self.logger.info(f"Очистка завершена. Всего удалено: {total_deleted} версий")
                
        except Exception as e:
            self.db_conn.rollback()
            self.logger.error(f"Ошибка при очистке версий моделей: {str(e)}")
            raise

    def close(self) -> None:
        """Корректное закрытие соединений."""
        if hasattr(self, 'db_conn') and self.db_conn:
            self.db_conn.close()
            self.logger.info("Соединение с БД закрыто")

def main():
    """Точка входа для CLI."""
    parser = argparse.ArgumentParser(
        description="Сервис управления базой данных для системы обучения моделей обнаружения аномалий"
    )
    parser.add_argument('--config', help="Путь к конфигурационному файлу (переопределяет USER_CONFIG_PATH)")
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Команда init
    init_parser = subparsers.add_parser('init', help='Инициализация БД')
    
    # Команда clean
    clean_parser = subparsers.add_parser('clean', help='Очистка старых версий моделей')
    
    args = parser.parse_args()
    
    # Проверяем обязательные переменные среды
    required_env_vars = [ENV_VICTORIAMETRICS_URL, ENV_DB_CONN_STRING]
    missing_vars = [var for var in required_env_vars if var not in os.environ]
    
    if missing_vars:
        print(f"Ошибка: отсутствуют обязательные переменные среды: {', '.join(missing_vars)}")
        exit(1)
    
    manager = None
    try:
        manager = MadTrainerDatabaseManager(args.config)
        
        if args.command == 'init':
            manager.init_database()
        elif args.command == 'clean':
            manager.clean_old_versions()
            
    except Exception as e:
        logging.error(f"Ошибка выполнения команды: {str(e)}", exc_info=True)
        exit(1)
    finally:
        if manager:
            manager.close()

if __name__ == "__main__":
    main()