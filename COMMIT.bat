@echo off
echo %~dp0
@REM git remote add all https://github.com/Alvaro-SP/PROYECTO-COMPI2.git
@REM git remote set-url --add --push all https://github.com/Alvaro-SP/Project2_OLC2VJ.git



            @REM git remote set-url --add --push origin https://Alvaro-SP:2f71685213f08a9e6bc04341bf58bca297c29fad@github.com/Alvaro-SP/PROYECTO-COMPI2.git
            @REM git remote set-url --add --push all https://github.com/Alvaro-SP/PROYECTO-COMPI2.git
            @REM git remote set-url --add --push origin https://github.com/Alvaro-SP/PROYECTO-COMPI2.git
@REM git pull https://Alvaro-SP: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@github.com/Alvaro-SP/Project2_OLC2VJ.git
@REM SET /P NOMBRE=WRITE COMMIT:
git add .
git commit -m "commit"
git push
git push https://Alvaro-SP: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@github.com/Alvaro-SP/Project2_OLC2VJ.git
pause


