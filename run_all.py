"""Запуск всех пяти заданий из любой рабочей папки."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for number in range(1, 6):
    print(f'\n========== Задание 1.{number} ==========\n', flush=True)
    subprocess.run([sys.executable, str(root/'code'/f'lr1_{number}.py'),
                    str(root/'tests'/f'lr1_{number}.txt')], check=True)
