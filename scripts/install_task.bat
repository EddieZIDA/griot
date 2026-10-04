@echo off
rem Cree (ou remplace) la tache planifiee Windows "Griot Ingestion" :
rem une ingestion toutes les 6 heures. A lancer une seule fois, par double-clic.
rem Pour la supprimer : schtasks /Delete /TN "Griot Ingestion" /F
schtasks /Create /TN "Griot Ingestion" /TR "\"%~dp0run_ingest.bat\"" /SC HOURLY /MO 6 /F
if errorlevel 1 (
  echo Echec de la creation de la tache.
) else (
  echo Tache creee. Journal : logs\ingest.log
)
pause
