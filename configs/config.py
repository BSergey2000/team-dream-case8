"""
Конфигурационный файл проекта.
"""

import logging  # Для дневника (лога)
from pathlib import Path  # Для путей
import sys  # Для добавления путей

# Главная папка проекта
ROOT_DIR = Path(__file__).parent.parent.resolve()

# Добавляем путь
sys.path.insert(0, str(ROOT_DIR))

# Папки проекта
DATA_DIR = ROOT_DIR / 'data'  # Для данных
RESULTS_DIR = ROOT_DIR / 'results'  # Для результатов
PLOTS_DIR = RESULTS_DIR / 'plots'  # Для графиков
LOGS_DIR = ROOT_DIR / 'logs'  # Для логов
MODELS_DIR = ROOT_DIR / 'models'  # Для моделей

# Создаём папки, если их нет
for dir_path in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR, MODELS_DIR]:
    try:
        # Создаём - parents, exist_ok — если уже есть
        dir_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise PermissionError(
            f"Нет прав создать папку: {dir_path}. Проверьте разрешения.")
    except Exception as e:
        raise RuntimeError(f"Ошибка создания папки {dir_path}: {e}.")

# Путь к данным
DATA_PATH = DATA_DIR / 'financials.csv'
# Если не найдена на этапе импорта, не прерываем импорт — просто логируем.
# Функции загрузки данных должны проверять существование файла и ругаться локально.
if not DATA_PATH.exists():
    logger = logging.getLogger(__name__)
    logger.warning(
        f"Данные не найдены по ожидаемому пути: {DATA_PATH}. Некоторые функции могут не работать, пока файл не появится.")

# Константы для работы, как правила игры.
TARGET = 'Market Cap'  # Что предсказываем, как цель.
# Фиксированное число для случайности, чтобы результаты были одинаковыми.
RANDOM_STATE = 42
TEST_SIZE = 0.2  # Доля для теста, как 20% на проверку.

# Настройка дневника (лога).
logger = logging.getLogger(__name__)  # Дневник для этого файла.
logger.setLevel(logging.INFO)  # Уровень: INFO — обычные записи.

# Формат: время - уровень - сообщение.
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Опциональная поддержка цветного вывода в консоли
try:
    import colorama
    colorama.init()
    _COLORAMA_AVAILABLE = True
except Exception:
    _COLORAMA_AVAILABLE = False


# Дневник в консоль, как экран
stream_handler = logging.StreamHandler()


class ColorFormatter(logging.Formatter):
    """
    Форматтер с раскраской сообщений в консоли по правилам:
    - ERROR/CRITICAL -> красный
    - Сообщения о сохранении модели (содержат 'модель сохранена' или 'saved') -> синий
    - INFO -> жёлтый
    - Остальные сообщения -> зелёный
    """
    RED = '\x1b[31m'
    YELLOW = '\x1b[33m'
    BLUE = '\x1b[34m'
    GREEN = '\x1b[32m'
    RESET = '\x1b[0m'

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        try:
            msg = record.getMessage()
        except Exception:
            msg = ''

        # Ошибки имеют абсолютный приоритет
        if record.levelno >= logging.ERROR:
            color = self.RED
        # Если сообщение явно сообщает о сохранении модели — делаем синим
        elif 'модель сохранена' in msg.lower() or 'saved' in msg.lower():
            color = self.BLUE
        elif record.levelno == logging.INFO:
            color = self.YELLOW
        else:
            color = self.GREEN

        # Если colorama доступна или терминал поддерживает ANSI — красим, иначе возвращаем plain
        if _COLORAMA_AVAILABLE or sys.platform != 'win32':
            return f"{color}{base}{self.RESET}"
        return base


stream_handler.setFormatter(ColorFormatter(
    '%(asctime)s - %(levelname)s - %(message)s'))

# Добавляем обработчики только если их ещё нет — это предотвращает дублирование логов при повторных импортах.
if not logger.handlers:
    logger.addHandler(stream_handler)

    # Дневник в файл
    try:
        # 'a' — добавлять, utf-8 — для кирилицы.
        file_handler = logging.FileHandler(
            LOGS_DIR / 'project.log', mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Дневник в файл готов: {LOGS_DIR / 'project.log'}")
    except Exception as e:
        logger.warning(
            f"Ошибка с файлом дневника: {e}. Вывод только на экран.")
