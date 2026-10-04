@echo off
rem Lance une ingestion Griot et ajoute la sortie a logs\ingest.log.
rem Appele par la tache planifiee (voir install_task.bat), ou a la main.
cd /d "%~dp0.."
if not exist logs mkdir logs
set PYTHONIOENCODING=utf-8
echo ==== %date% %time% ==== >> logs\ingest.log
".venv\Scripts\python.exe" ingest.py >> logs\ingest.log 2>&1
