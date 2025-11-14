@echo off
REM Run irrigation predictor with sensor data from MongoDB

REM ---- EDIT THESE IF NEEDED ----
SET MODEL_PATH=models\soil_moisture_pump_model.pkl
SET SCRIPT_DIR=%~dp0
SET MONGO_URI=mongodb://127.0.0.1:27017
REM --------------------------------

REM Optional: device-id as first argument
SET DEVICE_ID=%~1
REM Optional: mongo-uri as second argument
IF NOT "%~2"=="" SET MONGO_URI=%~2

echo Fetching latest sensor data from MongoDB...
echo.

IF "%~1"=="" (
    REM No device-id specified, fetch latest from any device
    python "%SCRIPT_DIR%src\predict.py" ^
        --model "%SCRIPT_DIR%%MODEL_PATH%" ^
        --from-db ^
        --mongo-uri "%MONGO_URI%"
) ELSE (
    REM Device-id specified, fetch latest from specific device
    python "%SCRIPT_DIR%src\predict.py" ^
        --model "%SCRIPT_DIR%%MODEL_PATH%" ^
        --from-db ^
        --device-id "%DEVICE_ID%" ^
        --mongo-uri "%MONGO_URI%"
)

pause
