
@REM cd "C:\Users\spuser\OneDrive - Facultad de Ingeniería de la Universidad de San Carlos de Guatemala\5.1 VAQUERAS JUNIO 2022\LAB COMPILADORES\PROYECTO-COMPI2"
@echo off
echo %~dp0
SET /P NOMBRE=WRITE COMMIT:
git add .
git commit -m "%NOMBRE%"
git push
git push https://Alvaro-SP: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@github.com/Alvaro-SP/Project2_OLC2VJ.git
@REM pause


