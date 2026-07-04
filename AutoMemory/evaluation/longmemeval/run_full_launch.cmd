@echo off
REM Launcher for the off-peak full run (Windows Task Scheduler).
REM Runs the 150-question 3-system pipeline (ingest -> automem -> rag -> none ->
REM judge) at the DeepSeek off-peak window. Each phase is resume-able, so the
REM whole launcher retries on a transient failure (e.g. a network blip) and the
REM ingest continues via --resume instead of restarting.
cd /d "%~dp0"
set PY=C:\Users\cjy24\AppData\Local\Programs\Python\Python313\python.exe
set LOG=..\results\lme_full_pipeline.log
echo [launcher] === start %DATE% %TIME% === >> "%LOG%"
for /L %%i in (1,1,4) do (
  "%PY%" run_full.py --tag lme_full --limit 150 --workers 8 --rag-workers 2 >> "%LOG%" 2>&1
  if not errorlevel 1 goto done
  echo [launcher] attempt %%i exited with error, retrying in 60s... >> "%LOG%"
  timeout /t 60 /nobreak >nul
)
:done
echo [launcher] === end %DATE% %TIME% === >> "%LOG%"
